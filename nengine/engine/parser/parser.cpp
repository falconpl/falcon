/*
   FALCON - The Falcon Programming Language.
   FILE: parser/parser.cpp

   Parser subsystem main class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 17:36:38 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falcon/syntaxerror.h"


#include <falcon/parser/parser.h>
#include <falcon/parser/lexer.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/state.h>
#include <falcon/codeerror.h>
#include <falcon/trace.h>
#include <falcon/genericerror.h>

#include <falcon/error.h>

#include "./parser_private.h"

namespace Falcon {
namespace Parsing {

Parser::Private::Private():
   m_nextTokenPos(0)
{
}

Parser::Private::~Private()
{
   clearTokens();
   
   // clear the lexers
   {
      LexerStack::iterator iter = m_lLexers.begin();
      while( iter != m_lLexers.end() )
      {
         delete *iter;
         ++iter;
      }
   }
}


void Parser::Private::clearTokens()
{
   TokenStack::iterator iter = m_tokenStack->begin();
   while( iter != m_tokenStack->end() )
   {
      delete *iter;
      ++iter;
   }
   m_tokenStack->clear();
}


Parser::Private::ParseFrame::~ParseFrame()
{
   TRACE2("Destroying ParseFrame at %p", this );
   /*Alternatives::iterator iter = m_candidates.begin();

   while( iter != m_candidates.end() )
   {
      delete *iter;
      ++iter;
   }
*/
   //m_candidates.clear();
}

//========================================================

Parser::Parser():
   T_EOF("EOF"),
   T_EOL("EOL"),
   T_Float("Float"),
   T_Int("Int"),
   T_Name("Name"),
   T_String("String"),
   m_ctx(0),
   m_bIsDone(false),
   m_bInteractive(0)
{
   _p = new Private;
}


Parser::~Parser()
{
   clearErrors();
   delete _p;
}


void Parser::addState( State& state )
{
   TRACE( "Parser::addState -- adding state '%s'", state.name().c_ize() );
   _p->m_states[state.name()] = &state;
}


void Parser::pushState( const String& name )
{
   TRACE( "Parser::pushState -- pushing state '%s'", name.c_ize() );

   Private::StateMap::const_iterator iter = _p->m_states.find( name );
   if( iter != _p->m_states.end() )
   {
      _p->m_lStates.push_back( iter->second );

      // set new proxy pointers
      Private::StateFrame& bf = _p->m_lStates.back();
      _p->m_tokenStack = &bf.m_tokenStack;
      _p->m_pframes = &bf.m_pframes;
      _p->m_pErrorFrames = &bf.m_pErrorFrames;

   }
   else
   {
      throw new CodeError( ErrorParam( e_state, __LINE__, __FILE__ ).extra(name) );
   }
}

void Parser::pushState( const String& name, Parser::StateFrameFunc cf, void* data )
{
   pushState( name );

   Private::StateFrame& bf = _p->m_lStates.back();
   bf.m_cbfunc = cf;
   bf.m_cbdata = data;
}


void Parser::popState()
{
   TRACE( "Parser::popState -- popping state", 0 );
   if ( _p->m_lStates.empty() )
   {
      throw new CodeError( ErrorParam( e_underflow, __LINE__, __FILE__ ).extra("Parser::popState") );
   }

   StateFrameFunc func = _p->m_lStates.back().m_cbfunc;
   void *cbdata = _p->m_lStates.back().m_cbdata;

   _p->m_lStates.pop_back();
   TRACE1( "Parser::popState -- now topmost state is '%s'", _p->m_lStates.back().m_state->name().c_ize() );

   // reset proxy pointers
   Private::StateFrame& bf = _p->m_lStates.back();
   _p->m_tokenStack = &bf.m_tokenStack;
   _p->m_pframes = &bf.m_pframes;
   _p->m_pErrorFrames = &bf.m_pErrorFrames;

   // execute the callback (?)
   if( func != 0 )
   {
      func( cbdata );
   }
}


bool Parser::parse( const String& mainState )
{
   TRACE( "Parser::parse -- invoked with '%s'", mainState.c_ize() );

   // Preliminary checks. We need a lexer and we need to have the required state.
   if( _p->m_lLexers.empty() )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, __FILE__ ).extra("Parser::parse - pushLexer") );
   }

   // Check if we have a lexer
   Private::StateMap::const_iterator iter = _p->m_states.find( mainState );
   if( iter != _p->m_states.end() )
   {
      _p->m_lStates.clear();
      _p->m_lStates.push_back( iter->second );
   }
   else
   {
      throw new CodeError( ErrorParam( e_state, __LINE__, __FILE__ ).extra(mainState) );
   }

   //==========================================
   // Ok, we can start -- initialize the parser
   //
   _p->m_lErrors.clear();
   _p->clearTokens();
   parserLoop();

   // at the end of the parser loop, the stack should be empty, or we missed something
   // -- exception: the EOF token may or may not be parsed.
   if( ! isComplete() )
   {
      syntaxError();
   }
   // If we have no error we succeeded.
   return _p->m_lErrors.empty();
}


bool Parser::isComplete() const
{
   return _p->m_tokenStack->empty() || _p->m_tokenStack->front()->token().id() == T_EOF.id();
}


bool Parser::hasErrors() const
{
   return ! _p->m_lErrors.empty();
}


GenericError* Parser::makeError() const
{
   if( _p->m_lErrors.empty() )
   {
      return 0;
   }

   GenericError* cerr = new GenericError(ErrorParam(e_syntax));
   Private::ErrorList::iterator iter = _p->m_lErrors.begin();
   while( iter != _p->m_lErrors.end() )
   {
      ErrorDef& def = *iter;

      String sExtra = def.sExtra;
      if( def.nOpenContext != 0 )
      {
         if( sExtra.size() != 0 )
            sExtra += " -- ";
         sExtra += "from line ";
         sExtra.N(def.nOpenContext);
      }
      
      SyntaxError* err = new SyntaxError( ErrorParam( def.nCode, def.nLine )
            .module(def.sUri)
            .extra(sExtra));
      cerr->appendSubError(err);
      ++iter;
   }
   
   return cerr;
}


void Parser::clearErrors()
{
   _p->m_lErrors.clear();
}


void Parser::clearTokens()
{
   if( _p->m_tokenStack->size() > 0 )
      simplify(_p->m_tokenStack->size(), 0);
}

int32 Parser::tokenCount()
{
   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;
   return _p->m_tokenStack->size() - nDepth;
}

int32 Parser::availTokens()
{
   return _p->m_tokenStack->size() - _p->m_nextTokenPos;
}

TokenInstance* Parser::getNextToken()
{
   if( _p->m_nextTokenPos >= _p->m_tokenStack->size() )
   {
      return 0;
   }

   return (*_p->m_tokenStack)[_p->m_nextTokenPos++];
}

void Parser::resetNextToken()
{
   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;
   _p->m_nextTokenPos = nDepth;
}


void Parser::enumerateErrors( Parser::errorEnumerator& enumerator ) const
{
   Private::ErrorList::const_iterator iter = _p->m_lErrors.begin();
   while( iter != _p->m_lErrors.end() )
   {
      const ErrorDef& def = *iter;
      
      if ( ! enumerator( def, ++iter == _p->m_lErrors.end() ) )
         break;
   }
}


const String& Parser::currentSource() const
{
   return _p->m_lLexers.back()->uri();
}

void Parser::setContext( void* ctx )
{
   m_ctx = ctx;
}


void Parser::pushLexer( Lexer* lexer )
{
   _p->m_lLexers.push_back( lexer );
}


void Parser::popLexer()
{
   if( _p->m_lLexers.empty() )
   {
      throw new CodeError( ErrorParam( e_underflow, __LINE__, __FILE__ ).extra("Parser::popLexer") );
   }

   delete _p->m_lLexers.back();
   _p->m_lLexers.pop_back();
}


void Parser::addError( int code, const String& uri, int l, int c, int ctx, const String& extra )
{
   TRACE( "Parser::addError -- with current stack: %s ", dumpStack().c_ize() );
   _p->m_lErrors.push_back(ErrorDef(code, uri, l, c, ctx, extra));
}


void Parser::addError( int code, const String& uri, int l, int c, int ctx  )
{
   TRACE( "Parser::addError -- with current stack: %s ", dumpStack().c_ize() );
   _p->m_lErrors.push_back(ErrorDef(code, uri, l, c, ctx));
}


void Parser::simplify( int32 tcount, TokenInstance* newtoken )
{
   TRACE( "Parser::simplify -- %d tokens -> %s",
         tcount, newtoken ? newtoken->token().name().c_ize() : "<nothing>" );

   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;

   if( tcount < 0 || tcount + nDepth > _p->m_tokenStack->size() )
   {
      throw new CodeError(ErrorParam(e_underflow, __LINE__, __FILE__ )
            .extra("Parser::simplify - tcount out of range"));
   }


   if( tcount != 0 )
   {
      size_t end = nDepth + tcount;
      for( size_t pos = nDepth; pos < end; ++pos )
      {
         delete (*_p->m_tokenStack)[pos];
      }

      _p->m_tokenStack->erase( _p->m_tokenStack->begin() + nDepth, _p->m_tokenStack->begin() + end );
   }

   if( newtoken != 0 )
   {
      _p->m_tokenStack->insert( _p->m_tokenStack->begin() + nDepth, newtoken );
   }
}

bool Parser::step()
{
   TRACE( "Parser::step", 0 );

   // Preliminary checks. We need a lexer and we need to have the required state.
   if( _p->m_lLexers.empty() )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, __FILE__ ).extra("Parser::step - pushLexer") );
   }

   // Check if we have a lexer
   if( _p->m_lStates.empty() )
   {
      throw new CodeError( ErrorParam( e_setup, __LINE__, __FILE__ ).extra("Parser::step - pushState") );
   }

   TRACE( "Parser::step -- on state \"%s\" -- %s ",
         _p->m_lStates.back().m_state->name().c_ize(), dumpStack().c_ize() );
   
   clearErrors();

   parserLoop();

   return ! hasErrors();
}



void Parser::syntaxError()
{
   TRACE( "Parser::syntaxError -- with current stack: %s ", dumpStack().c_ize() );

   String uri;
   int line = 0;
   int chr = 0;

   if( ! _p->m_lLexers.empty() )
   {
      uri = _p->m_lLexers.back()->uri();
   }

   if( ! _p->m_tokenStack->empty() )
   {
      line = _p->m_tokenStack->front()->line();
      chr = _p->m_tokenStack->front()->chr();
   }

   addError( e_syntax, uri, line, chr );

   _p->m_pframes->clear();
   size_t tc = tokenCount();
   if( tc > 0 )
   {
      simplify( tc, 0 );
   }
}


Parser::Path Parser::createPath() const
{
   return new Private::RulePath;
}


Parser::Path Parser::copyPath( Path p ) const
{
   return new Private::RulePath( *(Private::RulePath*) p );
}


void Parser::discardPath( Parser::Path p ) const
{
   delete (Private::RulePath*) p;
}


void Parser::confirmPath( Parser::Path p ) const
{
   //_p->m_pframes->back().m_candidates.push_back( (Private::RulePath*) p );
}


void Parser::addRuleToPath( Parser::Path p, Rule* r ) const
{
   Private::RulePath* path = ((Private::RulePath*) p);
   //path->push_front(r);
}


void Parser::addRuleToPath( const Rule* r ) const
{
   _p->m_pframes->back().m_path.push_back( r );
}

String Parser::dumpStack() const
{
   String sTokens;

   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;

   for( int nTokenLoop = 0; nTokenLoop < _p->m_tokenStack->size(); ++nTokenLoop )
   {
      if ( sTokens.size() > 0 )
      {
         sTokens += ", ";
      }

      if( nTokenLoop == nDepth )
      {
         sTokens += ">> ";
      }

      sTokens += (*_p->m_tokenStack)[nTokenLoop]->token().name();
   }

   return sTokens;
}


void Parser::addParseFrame( NonTerminal* token, int pos )
{
   if( pos < 0 )
   {
      pos = _p->m_tokenStack->size()-1;
   }
   
   _p->m_pframes->push_back(Private::ParseFrame(token,pos));
   resetNextToken();
}

size_t Parser::rulesDepth() const
{
   Private::ParseFrame& frame = _p->m_pframes->back();
   return frame.m_path.size();
}

size_t Parser::frameDepth() const
{
   return _p->m_pframes->size();
}

void Parser::unroll( size_t fd, size_t rd )
{
   _p->m_pframes->resize(fd);
   if( fd > 0 )
   {
      _p->m_pframes->back().m_path.resize(rd);
   }
}


void Parser::setFramePriority( const Token& t )
{
   Private::ParseFrame& frame = _p->m_pframes->back();
   if ( t.prio() != 0 && (frame.m_nPriority == 0 || t.prio() < frame.m_nPriority ) )
   {
      frame.m_nPriority = t.prio();
      // use the right-associativity of the strongest operator.
      frame.m_bIsRightAssoc = t.isRightAssoc();
      frame.m_prioFrame = _p->m_tokenStack->size();
   }
}


//==========================================
// Main parser algorithm.
//

void Parser::parserLoop()
{
   TRACE( "Parser::parserLoop -- starting", 0 );

   m_bIsDone = false;

   Lexer* lexer = _p->m_lLexers.back();
   while( ! m_bIsDone )
   {
      // we're done ?
      if( lexer == 0 )
      {
         TRACE( "Parser::parserLoop -- done on lexer pop", 0 );
         return;
      }

      TokenInstance* ti = lexer->nextToken();
      while( ti == 0 )
      {
         if( m_bInteractive )
         {
            TRACE( "Parser::parserLoop -- done on interactive lexer token shortage", 0 );
            return;
         }

         popLexer();
         if( _p->m_lLexers.empty() )
         {
            lexer = 0;
            break;
         }
         else
         {
            lexer = _p->m_lLexers.back();
            ti = lexer->nextToken();
         }
      }

      if( ti == 0 )
      {
         TRACE( "Parser::parserLoop -- Last loop with EOF as next", 0 );
         return;
         ti = new TokenInstance(0, 0, T_EOF );
      }

      _p->m_tokenStack->push_back(ti);

      TRACE( "Parser::parserLoop -- stack now: %s ", dumpStack().c_ize() );
      
      onNewToken();
   }

   TRACE( "Parser::parserLoop -- done on request", 0 );
}


void Parser::onNewToken()
{
   // If we don't have parsing frames, try to build new ones from the current state.
   if( _p->m_pframes->empty() )
   {
      TRACE("Parser::onNewToken -- starting new path finding", 0 );
      
      //... let the current state to find a path for us.
      State* curState = _p->m_lStates.back().m_state;

      // still empty?
      if( curState->findPaths( *this ) )
      {
         TRACE("Parser::onNewToken -- path found in current state.", 0 );
      }
      else
      {
         TRACE("Parser::onNewToken -- path NOT found.", 0 );
         // then, we have a syntax error
         syntaxError();
         return;
      }
   }
   else
   {
      // process existing frames.      
      if (! findPaths( true ) )
      {
          TRACE("Parser::onNewToken -- failed in incremental mode, exploring.", 0 );
         // may fail if incremental, try again in full mode.
         explorePaths();
         /*if ( ! findPaths( false ) )
         {
           
            parseError();
            return;
         }
          */
      }
   }

   // try to simplify the stack, if possible.
   applyPaths();
}


TokenInstance* Parser::getCurrentToken( int& pos ) const
{
   if ( _p->m_pframes->empty() || _p->m_tokenStack->empty() )
   {
      TRACE("Parser::getCurrentToken -- stack empty", 0 );
      return 0;
   }

   Private::ParseFrame& frame = _p->m_pframes->back();
   pos = _p->m_tokenStack->size();
   fassert( pos > 0 );
   pos--;

   TokenInstance* ret = (*_p->m_tokenStack)[pos];
   pos -= frame.m_nStackDepth;
   fassert( pos >= 0 );
   TRACE("Parser::getCurrentToken -- current token is at %d: %s", 
         pos, ret->token().name().c_ize() );

   return ret;
}

bool Parser::findPaths( bool bIncremental )
{
   Private::RulePath& path = _p->m_pframes->back().m_path;

   if( !path.empty() )
   {
      TRACE("Parser::findPaths -- continuing on existing path '%s'", path.back()->name().c_ize() );
      int nBaseFrames = frameDepth();
      int nBaseRules = rulesDepth();

      if( path.back()->match( *this, bIncremental ) )
      {
         return true;
      }

      unroll( nBaseFrames, nBaseRules );

      if ( ! bIncremental )
      {
         TRACE("Parser::findPaths -- existing path failed, trying with full on %s",
               _p->m_pframes->back().m_owningToken->name().c_ize() );

         path.clear();
         return _p->m_pframes->back().m_owningToken->findPaths(*this);
      }
      return false;
   }
   else
   {
      TRACE("Parser::findPaths -- no existing path, trying with full", 0 );
      return _p->m_pframes->back().m_owningToken->findPaths(*this);
   }
}


void Parser::parseError()
{
   TRACE("Parser::parseError -- raising now", 0 );

   // reverse the error frames, in case this error was detected
   // after a reverse-unroll loop
   while( !_p->m_pErrorFrames->empty() )
   {
      _p->m_pframes->push_back(_p->m_pErrorFrames->back());
      _p->m_pErrorFrames->pop_back();
   }
   
   // find an error handler in the current rule frame.
   // if not present, unroll the frame and try again
   // -- fall back to syntax error.
   while( ! _p->m_pframes->empty() )
   {
      Private::ParseFrame& frame = _p->m_pframes->back();

      // unroll the tokens in the stack.
      simplify( _p->m_tokenStack->size() - frame.m_nStackDepth, 0 );

      Private::RulePath::reverse_iterator iter = frame.m_path.rbegin();

      while( iter != frame.m_path.rend() )
      {
         const Rule* rule = *iter;
         if( rule->parent().errorHandler() != 0 )
         {
            _p->m_pframes->pop_back();
            TRACE("Parser::parseError -- applying error handler for %s",
                  rule->parent().name().c_ize() );
            
            rule->parent().errorHandler()( &rule->parent(), this );
            // we're done.
            return;
         }
         ++iter;
      }

      if( frame.m_owningToken->errorHandler() != 0 )
      {
         _p->m_pframes->pop_back();
         TRACE("Parser::parseError -- applying error handler for frame master '%s'",
                  frame.m_owningToken->name().c_ize() );

         frame.m_owningToken->errorHandler()( frame.m_owningToken, this );
         // we're done.
         return;
      }

      TRACE1("Parser::parseError -- error handler not found in current frame", 0);
      _p->m_pframes->pop_back();
   }

   TRACE1("Parser::parseError -- error handler not found", 0);
   syntaxError();
}


void Parser::applyPaths()
{
   TRACE("Parser::applyPaths -- begin", 0 );

   while( ! _p->m_pframes->empty() )
   {
      // get the deepest rule in the deepest frame.
      Private::ParseFrame& frame = _p->m_pframes->back();
      if( frame.m_path.empty() )
      {
         TRACE("Parser::applyPaths -- current frame path is empty, need more tokens.", 0 );
         return;
      }
      
      const Rule* currentRule = frame.m_path.back();
      int tcount = _p->m_tokenStack->size() - frame.m_nStackDepth;
      int rsize = currentRule->arity();

      // When arity is the same as tokens, we can simplify if we don't have
      // prioritized tokens.
      if (tcount == rsize )
      {
         if( currentRule->isGreedy() )
         {
            const NonTerminal* nt = static_cast<const NonTerminal*>(&_p->m_tokenStack->back()->token());
            addParseFrame(const_cast<NonTerminal*>(nt), _p->m_tokenStack->size()-1);
            
            // greedy rules always end with non-terminals
            TRACE("Parser::applyPaths -- same arity, descending on greedy rule '%s' in '%s' ",
                  currentRule->name().c_ize(), nt->name().c_ize());
            return;
         }
         else if( ((! frame.m_bIsRightAssoc) && frame.m_nPriority == 0)
            || !currentRule->getTokenAt(currentRule->arity()-1)->isNT() )
         {
            applyCurrentRule();
            TRACE("Parser::applyPaths -- Applied on same arity, stack: %s",
               dumpStack().c_ize() );
         }
         else
         {
            // else, we must wait for more tokens.
            TRACE("Parser::applyPaths -- Need more tokens (same arity), returning.", 0);
            return;
         }
      }

      // When we have more tokens than arity, then it means we have some priority
      // -- token in the stack
      else if( tcount > rsize )
      {
         // In this case, we must check for the associativity/prio of topmost token.
         const Token& next = _p->m_tokenStack->back()->token();
         //int tokenPrio = currentRule->isGreedy() ? 1 :
         int tokenPrio = next.prio();
         if( tokenPrio == 0 || tokenPrio > frame.m_nPriority
            || ( tokenPrio == frame.m_nPriority && ! frame.m_bIsRightAssoc )
            )
         {
            // we can simplify.
            applyCurrentRule();
            TRACE("Parser::applyPaths -- Applied on small arity, stack: %s",
               dumpStack().c_ize() );
         }
         else
         {
            TRACE("Parser::applyPaths -- small arity but considering prio/assoc", 0 );

            int frameDepth = frame.m_prioFrame; // better to cache it now
            int frameTokenPos = frameDepth - frame.m_nStackDepth;
            // now, we're interested in the non-terminal that was following the
            // operator.
            Token* tok = currentRule->getTokenAt(frameTokenPos);
            if( tok == 0 )
            {
               // rule exausted, but at lower priority
               // find the topmost nonterminal.
               while( frameDepth > frame.m_nStackDepth &&
                     !(*_p->m_tokenStack)[frameDepth]->token().isNT() )
               {
                  frameDepth --;
               }
               
               if( frameDepth == frame.m_nStackDepth ||
                     !(*_p->m_tokenStack)[frameDepth]->token().isNT() )
               {
                  TRACE("Parser::applyPaths -- rule exausted at a lower priority, but no alternative around.", 0);
                  applyCurrentRule();
                  TRACE("Parser::applyPaths -- stack now:", dumpStack().c_ize());
               }
               else
               {
                  TRACE("Parser::applyPaths -- rule exausted at a lower priority, putting frames forward.", 0);
                  const NonTerminal* nt = static_cast<const NonTerminal*>(&(*_p->m_tokenStack)[frameDepth]->token());
                  addParseFrame(const_cast<NonTerminal*>(nt), frameDepth);
                  TRACE("Parser::applyPaths -- stack now: %s", dumpStack().c_ize());
               }
            }
            else if ( tok != 0 && tok->isNT() )
            {
               TRACE("Parser::applyPaths -- small arity, descending into next token: %s",
                  tok->name().c_ize());

               NonTerminal* nt = static_cast<NonTerminal*>(tok);
               addParseFrame(nt, frameDepth);
            }
            // else proceed matching paths
         }
      }
      else 
      {
         // we simply don't have enough tokens.
         TRACE("Parser::applyPaths -- Need more tokens (larger arity %d > %d), returning.", rsize, tcount);
         return;
      }

      // if we're here, it means we applied at least one rule.
      TRACE("Parser::applyPaths -- Rule applied or added, looping again", 0);

      // now we must check if the effect of the reduction matches with the new
      // current rule, else we must either descend a find a matching rule or
      // declare failure.
      if( ! _p->m_pframes->empty() )
      {
         explorePaths();
      }
   }

   TRACE("Parser::applyPaths -- frames completelty exausted", 0 );
}


void Parser::explorePaths()
{
   while( ! _p->m_pframes->empty() && ! findPaths(false) )
   {
      _p->m_pErrorFrames->push_back(_p->m_pframes->back());
      _p->m_pframes->pop_back();
   }

   if ( _p->m_pframes->empty() )
   {
      parseError();
      return;
   }
   
   _p->m_pErrorFrames->clear();
}

void Parser::applyCurrentRule()
{
   Private::FrameStack* stack = _p->m_pframes;
   Private::ParseFrame& frame = stack->back();
   fassert( ! frame.m_path.empty() );
   const Rule* currentRule = frame.m_path.back();

   resetNextToken();
   currentRule->apply(*this);

   frame.m_path.pop_back();
   // eventually, pop the frame.
   if( frame.m_path.empty() )
   {
      stack->pop_back();
   }
}


void Parser::reset()
{
   resetNextToken();
   _p->clearTokens();
   _p->m_lStates.clear();
}

}
}

/* end of parser/parser.cpp */

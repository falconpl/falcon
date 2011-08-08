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

#define SRC "engine/parser/parser.cpp"

#include <falcon/syntaxerror.h>
#include <falcon/parser/parser.h>
#include <falcon/parser/lexer.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/state.h>
#include <falcon/codeerror.h>
#include <falcon/trace.h>
#include <falcon/error.h>
#include <falcon/genericerror.h>

#include "./parser_private.h"

namespace Falcon {
namespace Parsing {

Parser::Private::Private():
   m_nextTokenPos(0),
   m_stateFrameID(0)
{
}

Parser::Private::~Private()
{
   clearTokens();
   clearStates();

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
   if(m_tokenStack)
   {
      TokenStack::iterator iter = m_tokenStack->begin();
      while( iter != m_tokenStack->end() )
      {
         delete *iter;
         ++iter;
      }
      m_tokenStack->clear();
   }
}

void Parser::Private::clearStates()
{
   StateStack::iterator iter = m_lStates.begin();
   while( iter != m_lStates.end() )
   {
      delete *iter;
      ++iter;
   }
   m_lStates.clear();
   
   m_tokenStack = 0;
   m_pframes = 0;
   m_pErrorFrames = 0;
}


Parser::Private::ParseFrame::~ParseFrame()
{
   TRACE2("Destroying ParseFrame at %p", this );
}


Parser::Private::StateFrame::StateFrame( State* s ):
   m_state( s ),
   m_cbfunc( 0 ),
   m_cbdata( 0 ),
   m_id(0),
   m_appliedRules(0)
{
   TRACE2("Creating StateFrame at %p", this );
}

Parser::Private::StateFrame::~StateFrame()
{
   TRACE2("Destroying StateFrame at %p", this );
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
   m_bInteractive(0),
   m_consumeToken(0)
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
   TRACE2( "Parser::addState -- adding state '%s'", state.name().c_ize() );
   _p->m_states[state.name()] = &state;
}


void Parser::pushState( const String& name, bool isPushedState )
{
   TRACE1( "Parser::pushState -- pushing state '%s'", name.c_ize() );

   Private::StateMap::const_iterator iter = _p->m_states.find( name );
   if( iter != _p->m_states.end() )
   {
      if(!_p->m_lStates.empty())
      {
         TRACE("Parser::pushState -- pframes.size()=%d",(int)_p->m_pframes->size());
      }
      
      Private::StateFrame* stf = new Private::StateFrame( iter->second );
      _p->m_lStates.push_back( stf );

      // set new proxy pointers
      Private::StateFrame& bf = *stf;
      _p->m_tokenStack = &bf.m_tokenStack;
      _p->m_pframes = &bf.m_pframes;
      _p->m_pErrorFrames = &bf.m_pErrorFrames;
      bf.m_id = ++_p->m_stateFrameID;
      onPushState( isPushedState );
   }
   else
   {
      throw new CodeError( ErrorParam( e_state, __LINE__, __FILE__ ).extra(name) );
   }
}


void Parser::pushState( const String& name, Parser::StateFrameFunc cf, void* data )
{
   pushState( name );

   Private::StateFrame& bf = *_p->m_lStates.back();
   bf.m_cbfunc = cf;
   bf.m_cbdata = data;
}


void Parser::popState()
{
   MESSAGE2( "Parser::popState -- popping state" );
   if ( _p->m_lStates.size() < 2 )
   {
      throw new CodeError( ErrorParam( e_underflow, __LINE__, __FILE__ ).extra("Parser::popState") );
   }

   Private::StateFrame* sf = _p->m_lStates.back();
   
   StateFrameFunc func = sf->m_cbfunc;
   void *cbdata = sf->m_cbdata;

   // copy the residual tokens
   Private::TokenStack& ts = sf->m_tokenStack;
   Private::TokenStack& tsPrev = (*(++_p->m_lStates.rbegin()))->m_tokenStack;
   Private::TokenStack::iterator tsiter = ts.begin();
   while( tsiter != ts.end() )
   {
      tsPrev.push_back( *tsiter );
      ++tsiter;
   }
   ts.clear(); // prevent destruction of the tokens.
   
   
   _p->m_lStates.pop_back();
   delete sf;
   TRACE1( "Parser::popState -- now topmost state is '%s'", _p->m_lStates.back()->m_state->name().c_ize() );

   // reset proxy pointers
   Private::StateFrame& bf = *_p->m_lStates.back();
   _p->m_tokenStack = &bf.m_tokenStack;
   _p->m_pframes = &bf.m_pframes;
   _p->m_pErrorFrames = &bf.m_pErrorFrames;
   TRACE1("Parser::popState -- pframes.size()=%d, stack now: %s",
      (int)_p->m_pframes->size(), dumpStack().c_ize());

   // execute the callback (?)
   if( func != 0 )
   {
      func( cbdata );
   }

   // pop the last applied rule?
   while( bf.m_appliedRules > 0 )
   {
      fassert( !  bf.m_pframes.back().m_path.empty() );
      // the rule will be applied, so it's not in the path anymore.
      bf.m_pframes.back().m_path.pop_back();
      // eventually, pop the frame.
      if( bf.m_pframes.back().m_path.empty() )
      {
         bf.m_pframes.pop_back();
      }
      bf.m_appliedRules--;
   }

   onPopState();
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
      _p->clearStates();
      pushState( mainState, false );
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
   TRACE1( "Parser::isComplete? -- ptr %p", _p->m_tokenStack );
   if( _p->m_tokenStack == 0 )
   {
      return true;
   }
   
   TRACE1( "Parser::isComplete? -- %s", _p->m_tokenStack->empty() ? "empty" : "not empty");
   
   if( _p->m_tokenStack->empty() )
   {
      return true;
   }
   
   TRACE1( "Parser::isComplete? -- %s", 
         _p->m_tokenStack->front()->token().id() == T_EOF.id() ? "EOF" : "not eof" );
   
   return _p->m_tokenStack->front()->token().id() == T_EOF.id();
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
      if( def.nOpenContext != 0 && def.nOpenContext != def.nLine )
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


void Parser::clearFrames()
{
   _p->m_pframes->clear();
   size_t tc = tokenCount();
   if( tc > 0 )
   {
      simplify( tc, 0 );
   }
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

TokenInstance* Parser::getLastToken()
{
   if( _p->m_tokenStack->empty() )
   {
      return 0;
   }

   return _p->m_tokenStack->back();
}


void Parser::trimFromCurrentToken()
{
   if ( _p->m_nextTokenPos <  _p->m_tokenStack->size() )
   {
      _p->m_tokenStack->resize( _p->m_nextTokenPos );
   }
}

void Parser::resetNextToken()
{
   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;
   _p->m_nextTokenPos = nDepth;
}


void Parser::enumerateErrors( Parser::ErrorEnumerator& enumerator ) const
{
   Private::ErrorList::const_iterator iter = _p->m_lErrors.begin();
   while( iter != _p->m_lErrors.end() )
   {
      const ErrorDef& def = *iter;

      if ( ! enumerator( def, ++iter == _p->m_lErrors.end() ) )
         break;
   }
}


int32 Parser::lastErrorLine() const
{
   if( _p->m_lErrors.empty() )
   {
      return 0;
   }
   
   return _p->m_lErrors.back().nLine;
}


const String& Parser::currentSource() const
{
   return _p->m_lLexers.back()->uri();
}

int Parser::currentLine() const
{
   return _p->m_lLexers.back()->line();
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
   TRACE1( "Parser::simplify -- %d tokens -> %s",
         tcount, newtoken ? newtoken->token().name().c_ize() : "<nothing>" );

   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;

   if( tcount < 0 || tcount + nDepth > (int32) _p->m_tokenStack->size() )
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
   MESSAGE2( "Parser::step" );

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

   TRACE1( "Parser::step -- on state \"%s\" -- %s ",
         _p->m_lStates.back()->m_state->name().c_ize(), dumpStack().c_ize() );

   clearErrors();

   parserLoop();

   return ! hasErrors();
}



void Parser::syntaxError()
{
   TRACE1( "Parser::syntaxError -- with current stack: %s ", dumpStack().c_ize() );

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

   clearFrames();
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


void Parser::confirmPath( Parser::Path ) const
{
   //TODO Remove
   //_p->m_pframes->back().m_candidates.push_back( (Private::RulePath*) p );
}


void Parser::addRuleToPath( Parser::Path , Rule*  ) const
{
   //TODO Remove
   //Private::RulePath* path = ((Private::RulePath*) p);
   //path->push_front(r);
}


void Parser::addRuleToPath( const Rule* r ) const
{
   _p->m_pframes->back().m_path.push_back( r );
}

String Parser::dumpStack() const
{
   String sTokens;

   size_t nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;

   for( size_t nTokenLoop = 0; nTokenLoop < _p->m_tokenStack->size(); ++nTokenLoop )
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


void Parser::addParseFrame( const NonTerminal* token, int pos )
{
   TRACE1("Parser::addParseFrame -- %s at %d",token->name().c_ize(),pos);
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
   MESSAGE1( "Parser::parserLoop -- starting" );

   m_bIsDone = false;

   Lexer* lexer = _p->m_lLexers.back();
   while( ! m_bIsDone )
   {
      // we're done ?
      if( lexer == 0 )
      {
         MESSAGE2( "Parser::parserLoop -- done on lexer pop" );
         return;
      }

      TokenInstance* ti = lexer->nextToken();
      while( ti == 0 )
      {
         if( m_bInteractive )
         {
            MESSAGE2( "Parser::parserLoop -- done on interactive lexer token shortage" );
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
         MESSAGE2( "Parser::parserLoop -- Last loop with EOF as next" );
         return;
         ti = new TokenInstance(0, 0, T_EOF );
      }
      
      if( m_consumeToken != 0 )
      {
         TRACE1( "Parser::parserLoop -- Discarding token %s in search of %s", 
            ti->token().name().c_ize(), m_consumeToken->name().c_ize() );
         if ( ti->token().id() == m_consumeToken->id() )
         {
            m_consumeToken = 0;
         }
         continue;
      }
      else
      {
         _p->m_tokenStack->push_back(ti);
      }

      TRACE1( "Parser::parserLoop -- stack now: %s ", dumpStack().c_ize() );

      onNewToken();
   }

   MESSAGE2( "Parser::parserLoop -- done on request" );
}


void Parser::onNewToken()
{
   // If we don't have parsing frames, try to build new ones from the current state.
   if( _p->m_pframes->empty() )
   {
      MESSAGE1( "Parser::onNewToken -- starting new path finding" );

      //... let the current state to find a path for us.
      State* curState = _p->m_lStates.back()->m_state;

      // still empty?
      if( curState->findPaths( *this ) )
      {
         MESSAGE2( "Parser::onNewToken -- path found in current state." );
      }
      else
      {
         MESSAGE2( "Parser::onNewToken -- path NOT found." );
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
         MESSAGE2( "Parser::onNewToken -- failed in incremental mode, exploring." );
         // may fail if incremental, try again in full mode.
         explorePaths();
      }
   }

   // try to simplify the stack, if possible.
   applyPaths();
}


TokenInstance* Parser::getCurrentToken( int& pos ) const
{
   if ( _p->m_pframes->empty() || _p->m_tokenStack->empty() )
   {
      MESSAGE2( "Parser::getCurrentToken -- stack empty" );
      return 0;
   }

   Private::ParseFrame& frame = _p->m_pframes->back();
   pos = _p->m_tokenStack->size();
   fassert( pos > 0 );
   pos--;

   TokenInstance* ret = (*_p->m_tokenStack)[pos];
   pos -= frame.m_nStackDepth;
   fassert( pos >= 0 );
   TRACE1("Parser::getCurrentToken -- current token is at %d: %s",
         pos, ret->token().name().c_ize() );

   return ret;
}

bool Parser::findPaths( bool bIncremental )
{
   Private::RulePath& path = _p->m_pframes->back().m_path;

   if( !path.empty() )
   {
      TRACE2("Parser::findPaths -- continuing on existing path '%s'", path.back()->name().c_ize() );
      int nBaseFrames = frameDepth();
      int nBaseRules = rulesDepth();

      if( path.back()->match( *this, bIncremental ) )
      {
         TRACE1("Parser::findPaths -- path '%s' match", path.back()->name().c_ize() );
         return true;
      }

      unroll( nBaseFrames, nBaseRules );

      if ( ! bIncremental )
      {
         TRACE1("Parser::findPaths -- existing path failed, trying with full on %s",
               _p->m_pframes->back().m_owningToken->name().c_ize() );

         path.clear();
         bool bres = _p->m_pframes->back().m_owningToken->findPaths(*this);

         TRACE1("Parser::findPaths -- '%s'->findPaths %s",
               _p->m_pframes->back().m_owningToken->name().c_ize(), bres ? "success" : "failure" );
         return bres;
      }

      TRACE1("Parser::findPaths -- Existing path '%s' failed ", path.back()->name().c_ize() );
      return false;
   }
   else
   {
      MESSAGE2( "Parser::findPaths -- no existing path, trying with full" );
      return _p->m_pframes->back().m_owningToken->findPaths(*this);
   }
}


void Parser::parseError()
{
   MESSAGE1( "Parser::parseError -- raising now" );

   // reverse the error frames, in case this error was detected
   // after a reverse-unroll loop
   while( !_p->m_pErrorFrames->empty() )
   {
      _p->m_pframes->push_back(_p->m_pErrorFrames->back());
      _p->m_pErrorFrames->pop_back();
   }
   

   //_p->m_pframes->clear();

   // find an error handler in the current rule frame.
   // if not present, unroll the frame and try again
   // -- fall back to syntax error.
   while( ! _p->m_pframes->empty() )
   {
      Private::ParseFrame& frame = _p->m_pframes->back();      

      while( ! frame.m_path.empty() )
      {
         const Rule* rule = frame.m_path.back();
         if( rule->parent().errorHandler() != 0 )
         {
            TRACE2("Parser::parseError -- applying error handler for %s",
                  rule->parent().name().c_ize() );

            resetNextToken();
            // we're done.
            if( rule->parent().errorHandler()( rule->parent(), *this ) )
            {
               // handled?
               return;
            }
         }
         frame.m_path.pop_back();
      }

      if( frame.m_owningToken->errorHandler() != 0 )
      {
         TRACE2("Parser::parseError -- applying error handler for frame master '%s'",
                  frame.m_owningToken->name().c_ize() );

         resetNextToken();
         // we're done.
         if( frame.m_owningToken->errorHandler()( *frame.m_owningToken, *this ) )
         {
            // handled?
            //_p->m_pframes->pop_back();
            return;
         }
      }

      MESSAGE2( "Parser::parseError -- error handler not found in current frame" );
      _p->m_pframes->pop_back();
   }

   // Then, scan the CURRENT STACK backward, and try to fix the latest error.
   MESSAGE2( "Parser::parseError -- No handler frame found, trying on the stack." );
   Private::TokenStack::reverse_iterator riter = _p->m_tokenStack->rbegin();
   int pos = _p->m_tokenStack->size()-1;
   while( riter != _p->m_tokenStack->rend() )
   {
      TokenInstance* ti = *riter;
      if( ti->token().isNT() )
      {
         const NonTerminal* nt = static_cast<const NonTerminal*>( &ti->token() );
         if( nt->errorHandler() != 0 )
         {
            // we found it.
            addParseFrame( nt, pos );
            TRACE1( "Parser::parseError -- stack now %s", dumpStack().c_ize() );
            if( nt->errorHandler()( *nt, *this ) )
            {
               // handled
               _p->m_pframes->pop_back();
               return;
            }
            MESSAGE( "Parser::parseError -- handler refused to work" );
            // nope, try again
            _p->m_pframes->pop_back();
         }
      }
      --pos;
      ++riter;
   }

   MESSAGE2( "Parser::parseError -- error handler not found" );
   syntaxError();
}


bool Parser::applyPaths()
{
   MESSAGE1( "Parser::applyPaths -- begin" );

   bool bLooped = false;

   while( ! _p->m_pframes->empty() )
   {
      bLooped = true;

      // get the deepest rule in the deepest frame.
      Private::ParseFrame& frame = _p->m_pframes->back();
      if( frame.m_path.empty() )
      {
         MESSAGE2( "Parser::applyPaths -- current frame path is empty, need more tokens." );
         return false;
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
            TRACE2("Parser::applyPaths -- same arity, descending on greedy rule '%s' in '%s' ",
                  currentRule->name().c_ize(), nt->name().c_ize());
            return false;
         }
         else if( ((! frame.m_bIsRightAssoc) && frame.m_nPriority == 0)
            || !currentRule->getTokenAt(currentRule->arity()-1)->isNT() )
         {
            applyCurrentRule();
            TRACE2("Parser::applyPaths -- Applied on same arity, stack: %s",
               dumpStack().c_ize() );
         }
         else
         {
            // else, we must wait for more tokens.
            MESSAGE2( "Parser::applyPaths -- Need more tokens (same arity), returning." );
            return false;
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
            TRACE2("Parser::applyPaths -- Applied on small arity, stack: %s",
               dumpStack().c_ize() );
         }
         else
         {
            MESSAGE2( "Parser::applyPaths -- small arity but considering prio/assoc" );

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
                  MESSAGE2( "Parser::applyPaths -- rule exausted at a lower priority, but no alternative around." );
                  applyCurrentRule();
                  TRACE1("Parser::applyPaths -- applied with no alternives; stack now: %s", dumpStack().c_ize());
               }
               else
               {
                  MESSAGE2( "Parser::applyPaths -- rule exausted at a lower priority, putting frames forward." );
                  const NonTerminal* nt = static_cast<const NonTerminal*>(&(*_p->m_tokenStack)[frameDepth]->token());
                  addParseFrame(const_cast<NonTerminal*>(nt), frameDepth);
                  TRACE1("Parser::applyPaths -- applied with forward frame; stack now: %s", dumpStack().c_ize());
               }
            }
            else if ( tok != 0 && tok->isNT() )
            {
               TRACE2("Parser::applyPaths -- small arity, descending into next token: %s",
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
         TRACE1("Parser::applyPaths -- Need more tokens (larger arity %d > %d), returning.", rsize, tcount);
         return false;
      }

      // if we're here, it means we applied at least one rule.
      MESSAGE2( "Parser::applyPaths -- Rule applied or added, looping again" );

      // now we must check if the effect of the reduction matches with the new
      // current rule, else we must either descend a find a matching rule or
      // declare failure.
      if( ! _p->m_pframes->empty() && ! _p->m_tokenStack->empty() )
      {
         explorePaths();
      }
   }

   TRACE1("Parser::applyPaths -- frames completelty exausted (%s)",
         bLooped ? "Having worked" : "Did nothing" );
   return bLooped;
}


void Parser::explorePaths()
{
   MESSAGE1( "Parser::explorePaths -- starting full search mode." );

   while( ! _p->m_pframes->empty() && ! findPaths(false) )
   {
      TRACE1( "Parser::explorePaths -- adding error frame %d(%s); parser frames %d",
         (int) _p->m_pErrorFrames->size(),
         //_p->m_pErrorFrames->empty() ? "none" : _p->m_pErrorFrames->back().m_path.back()->name().c_ize(),
         _p->m_pErrorFrames->empty() ? "none" : _p->m_pErrorFrames->back().m_owningToken->name().c_ize(),
         (int) _p->m_pframes->size()
         );
      // failed on the same rule?
      if( ! _p->m_pErrorFrames->empty() &&
          _p->m_pframes->back().m_owningToken == _p->m_pErrorFrames->back().m_owningToken )
      {
         // we did loop.
         TRACE1( "Parser::explorePaths -- repeated error frame on '%s' -- aborting.",
               _p->m_pframes->back().m_owningToken->name().c_ize() );
         parseError();
         return;
      }

      _p->m_pErrorFrames->push_back(_p->m_pframes->back());
      _p->m_pframes->pop_back();
   }

   if ( _p->m_pframes->empty() )
   {
      parseError();
      return;
   }

   MESSAGE1( "Parser::explorePaths -- clearing errors." );
   //_p->m_pErrorFrames->clear();
}


void Parser::applyCurrentRule()
{
   Private::StateFrame& state = *_p->m_lStates.back();
   Private::FrameStack* stack = _p->m_pframes;
   Private::ParseFrame& frame = stack->back();
   fassert( ! frame.m_path.empty() );
   const Rule* currentRule = frame.m_path.back();
   int frameId = state.m_id;

   // If we apply, we know we have cleared all errors.
   _p->m_pErrorFrames->clear();
   resetNextToken();
   TRACE1( "Applying rule %s -- state depth %d -- state id %d",
      currentRule->name().c_ize(),
      (int) _p->m_lStates.size(),
      frameId );

   state.m_appliedRules++;
   currentRule->apply(*this);   

   TRACE3( "Applied rule %s -- state depth %d -- state id %d",
      currentRule->name().c_ize(),
      (int) _p->m_lStates.size(),
      _p->m_lStates.back()->m_id );
   // did we changed state?
   if( frameId != _p->m_lStates.back()->m_id )
   {
      TRACE3( "Rule %s detect pop-state", currentRule->name().c_ize() );
      return;
   }

   state.m_appliedRules--;
   // the rule will be applied, so it's not in the path anymore.
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
   //_p->clearTokens();
   _p->clearStates();
}


Lexer* Parser::currentLexer() const
{
   if( _p->m_lLexers.empty() )
   {
      return 0;
   }

   return _p->m_lLexers.back();
}

}
}

/* end of parser/parser.cpp */

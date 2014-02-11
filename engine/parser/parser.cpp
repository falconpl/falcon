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

#include <falcon/parser/parser.h>
#include <falcon/parser/lexer.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/state.h>
#include <falcon/trace.h>

#include <falcon/error.h>
#include <falcon/stderrors.h>

#include "./parser_private.h"

namespace Falcon {
namespace Parsing {

Parser::Private::Private():
   m_nextTokenPos(0),
   m_tokenStack(0),
   m_stateFrameID(0),
   m_lastLine(0)

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
         (*iter)->dispose();
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
   //TRACE2("Destroying ParseFrame at %p", this );
}


Parser::Private::StateFrame::StateFrame( NonTerminal* s ):
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
   T_DummyTerminal("*-.-*"),

   m_ctx(0),
   m_bInteractive(false),
   m_consumeToken(0),
   m_lastLine(0)
{
   _p = new Private;
   T_DummyTerminal.id(0);
}


Parser::~Parser()
{
   clearErrors();
   delete _p;
}


void Parser::addState( NonTerminal& nt )
{
   TRACE1( "Parser::addState(\"%s\")", nt.name().c_ize() );
   _p->m_states[nt.name()] = &nt;
}


void Parser::pushState( const String& name, bool isPushedState )
{
   TRACE( "Parser::pushState(\"%s\", %s)", name.c_ize(), isPushedState ? "true" : "false" );

   Private::StateMap::const_iterator iter = _p->m_states.find( name );
   if( iter != _p->m_states.end() )
   {
      if(!_p->m_lStates.empty())
      {
         TRACE1("Parser::pushState -- pframes.size()=%d",(int)_p->m_pframes->size());
      }

      NonTerminal* state = iter->second;
      Private::StateFrame* stf = new Private::StateFrame( state );
      _p->m_lStates.push_back( stf );

      // set new proxy pointers
      Private::StateFrame& bf = *stf;
      _p->m_tokenStack = &bf.m_tokenStack;
      _p->m_pframes = &bf.m_pframes;
      _p->m_pframes->push_back(Private::ParseFrame(state,0));
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
     //TODO
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

   GenericError* cerr = new GenericError(ErrorParam(e_compile, __LINE__, SRC ));
   Private::ErrorList::iterator iter = _p->m_lErrors.begin();
   while( iter != _p->m_lErrors.end() )
   {
      ErrorDef& def = *iter;
      if( def.objError != 0 )
      {
         cerr->appendSubError(def.objError);
      }
      else {
         String sExtra = def.sExtra;
         if( def.nOpenContext != 0 && def.nOpenContext != def.nLine )
         {
            if( sExtra.size() != 0 )
               sExtra += " -- ";
            sExtra += "from line ";
            sExtra.N(def.nOpenContext);
         }

         SyntaxError* err = new SyntaxError( ErrorParam( def.nCode )
               .module(def.sUri)
               .line(def.nLine)
               //.chr(def.nChar)
               .extra(sExtra));
         cerr->appendSubError(err);
      }

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
   _p->m_pframes->push_back(Private::ParseFrame(_p->m_lStates.back()->m_state,0));
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

void Parser::trimFromBase(unsigned int base, unsigned int count)
{
   if ( base+count > _p->m_tokenStack->size() )
   {
      if( base >= _p->m_tokenStack->size() )
      {
         return;
      }

      count = _p->m_tokenStack->size()-base;
   }

   int nDepth = _p->m_pframes->empty() ? 0 : _p->m_pframes->back().m_nStackDepth;

   size_t end = nDepth + count + base;
   for( size_t pos = nDepth+base; pos < end; ++pos )
   {
      (*_p->m_tokenStack)[pos]->dispose();
   }

   _p->m_tokenStack->erase( _p->m_tokenStack->begin() + nDepth+base, _p->m_tokenStack->begin() + end );
}

void Parser::trim( unsigned int count )
{
   if ( count > _p->m_tokenStack->size() )
   {
      count = _p->m_tokenStack->size();
   }

   _p->m_tokenStack->resize( _p->m_tokenStack->size() - count );
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

      if ( ! enumerator( def ) )
         break;
      ++iter;
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

   m_lastSource = _p->m_lLexers.back()->uri();
   m_lastLine = _p->m_lLexers.back()->line();
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


void Parser::addError( Error* error )
{
   TRACE( "Parser::addError -- with current stack: %s ", dumpStack().c_ize() );
   _p->m_lErrors.push_back(ErrorDef(error));
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
         (*_p->m_tokenStack)[pos]->dispose();
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
   consumeUpTo(T_EOL);
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


void Parser::pushParseFrame( const NonTerminal* token, int pos )
{
   //TRACE1("Parser::addParseFrame -- %s at %d",token->name().c_ize(),pos);
   if( pos < 0 )
   {
      pos = _p->m_tokenStack->size();
   }

   _p->m_pframes->push_back(Private::ParseFrame(token,pos));
   resetNextToken();
}


void Parser::popParseFrame()
{
   if( _p->m_pframes->size() > 1 )
   {
      _p->m_pframes->pop_back();
      resetNextToken();
   }
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

void Parser::parseError()
{
   MESSAGE( "Parser::parseError -- raising now" );

   if( _p->m_pErrorFrames->empty() )
   {
      syntaxError();
   }
   else
   {
      *_p->m_pframes = *_p->m_pErrorFrames;
      _p->m_pErrorFrames->clear();
      Private::ParseFrame& curf =  _p->m_pframes->back();
      TRACE1( "Parser::parserLoop -- using error handler for %s from position %d/%d.",
               curf.m_owningToken->name().c_ize(),
               curf.m_hypToken, curf.m_hypotesis );
      fassert( curf.m_owningToken->errorHandler() != 0 );
      curf.m_owningToken->errorHandler()( *curf.m_owningToken, *this );
   }

}


void Parser::saveErrorFrame()
{

   Private::ParseFrame& pf = _p->m_pframes->back();
   if( pf.m_bErrorMode )
   {
      TRACE( "Parser::saveErrorFrame -- Not saving Frame %d for rule %s(%d:%d) because in error mode.",
               _p->m_pframes->size(), pf.m_owningToken->name().c_ize(), pf.m_hypotesis, pf.m_hypToken );
      return;
   }

   if ( pf.m_owningToken->errorHandler() != 0 && pf.m_hypToken > 0 )
   {
      if ( _p->m_pErrorFrames->size() < _p->m_pframes->size() )
      {
         *_p->m_pErrorFrames = *_p->m_pframes;
         // reset the tokens, we'll have to scan them again
         Private::FrameStack::iterator iter = _p->m_pErrorFrames->begin();
         while( iter != _p->m_pErrorFrames->end() )
         {
            Private::ParseFrame& pf = *iter;
            pf.m_hypotesis = 0;
            pf.m_hypToken = 0;
            ++iter;
         }


         TRACE( "Parser::saveErrorFrame -- saving error frame of depth %d at rule %s",
                  _p->m_pErrorFrames->size(),
                  _p->m_pErrorFrames->back().m_owningToken->name().c_ize() );
      }
      else {
         TRACE( "Parser::saveErrorFrame -- Shallow error frame of depth %d at rule %s(%d:%d) not saved. ",
                  _p->m_pframes->size(), pf.m_owningToken->name().c_ize(), pf.m_hypotesis, pf.m_hypToken );
      }
   }
   else
   {
      TRACE( "Parser::saveErrorFrame -- Frame %d for rule %s(%d:%d) has no relevant error frame.",
               _p->m_pframes->size(), pf.m_owningToken->name().c_ize(), pf.m_hypotesis, pf.m_hypToken );
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

//========================================================================================================================
// Main parser algorithm.
//

void Parser::parserLoop()
{
   MESSAGE( "Parser::parserLoop -- starting" );

   m_bEOLGiven = false;
   while(true)
   {
      if( _p->m_tokenStack->empty() && ! readNextToken() )
      {
         MESSAGE( "Parser::parserLoop -- end on no more tokens" );
         break;
      }

      // a special case. We have the toplevel state in the stack.
      if( _p->m_tokenStack->size() == 1 && _p->m_lStates.front()->m_state == &_p->m_tokenStack->back()->token() )
      {
         MESSAGE1( "Parser::parserLoop -- Top token recognized." );
         simplify(1);
         if( m_bInteractive )
         {
            MESSAGE( "Parser::parserLoop -- Exit on interactive parsing of top token." );
            return;
         }
         continue;
      }

      // start where we left.
      Private::ParseFrame* currentFrame = &_p->m_pframes->back();
      const NonTerminal* current = currentFrame->m_owningToken;

      int32& hyp = currentFrame->m_hypotesis;
      int32& rulePos = currentFrame->m_hypToken;
      int32 stackPos = rulePos + currentFrame->m_nStackDepth;
      Token* rule = current->term(hyp);
      int32 ruleArity = rule->arity();
      const Token* ruleTok;
      const Token* stackTok;

      TRACE1( "Parser::parserLoop -- %s(%d:%s) with stack: %s",
                  currentFrame->m_owningToken->name().c_ize(), currentFrame->m_hypotesis, rule->name().c_ize(), dumpStack().c_ize() );

      while( rulePos < ruleArity
             && stackPos < (int) _p->m_tokenStack->size() )
      {
         ruleTok = rule->term(rulePos);
         stackTok = &(*_p->m_tokenStack)[stackPos]->token();
         if( ruleTok != stackTok )
         {
            break;
         }

         rulePos++;
         stackPos++;
         // are we traversing a priority?
         if( ruleTok->prio() > 0 && (currentFrame->m_prio == 0 || ruleTok->prio() < currentFrame->m_prio) )
         {
            currentFrame->m_prio = ruleTok->prio();
            currentFrame->m_prioPos = rulePos; // we record the next position.
            currentFrame->m_bRA = ruleTok->isRightAssoc();
            TRACE2( "Parser::parserLoop -- Frame priority %d%s at position %d",
                     currentFrame->m_prio, currentFrame->m_bRA? "(ra)" : "",  currentFrame->m_prioPos );
         }
      }


      // Is the rule complete?
      if( rulePos == ruleArity )
      {
         // then, if the rule ends with a terminal, we match.
         if( ! ruleTok->isNT() || (rulePos == 1 && ! static_cast<const NonTerminal*>(ruleTok)->isRecursive()) )
         {
            MESSAGE1( "Parser::parserLoop -- Matched: same arity and last token is terminal." );
            applyCurrentRule();
         }
         else {
            // if the stack is exausted, we need to ask for more.
            if( stackPos == (int) _p->m_tokenStack->size() )
            {
               MESSAGE1( "Parser::parserLoop -- Stack exausted and full match" );
               if( ! readNextToken() )
               {
                  MESSAGE( "Parser::parserLoop -- end on no more tokens" );
                  break;
               }
            }
            else {
               // we have at least 1 more token on the stack.
               stackTok = &(*_p->m_tokenStack)[stackPos]->token();
               TRACE2( "Parser::parserLoop -- Priority check: rule %d%s vs. next token %d%s",
                        stackTok->prio(), stackTok->isRightAssoc() ? "(ra)": "",
                        currentFrame->m_prio, currentFrame->m_bRA ? "(ra)": "");

               // if it has no priority, or lower priority, or same priority with left-assoc, we're done.
               if( stackTok->prio() == 0
                  || currentFrame->m_prio == 0
                  || stackTok->prio() > currentFrame->m_prio
                  || (!stackTok->isRightAssoc() && stackTok->prio() == currentFrame->m_prio))
               {
                  MESSAGE1( "Parser::parserLoop -- Matched: same arity and next token in stack is lower priority." );
                  applyCurrentRule();
               }
               // otherwise, we have to try again with the next operator.
               else {
                  MESSAGE1( "Parser::parserLoop -- Cheching higher priority operator." );
                  // don't abandon this hypotesis
                  pushParseFrame(current, currentFrame->m_prioPos + currentFrame->m_nStackDepth);
               }
            } // stack exausted
         } // nonterminal token
      } // complete scan

      else if( stackPos >= (int) _p->m_tokenStack->size() )
      {
         // we need to pull more tokens
         MESSAGE1( "Parser::parserLoop -- Pull more tokens");
         if( ! readNextToken() )
         {
            MESSAGE1( "Parser::parserLoop -- End on no more tokens" );
            break;
         }
      }
      // else, the match failed somewhere.
      else {
         //activate error recovery if necessary
         saveErrorFrame();

         // but if the token is non-terminal, it may match a sub-rule.
         int subRule = rulePos;
         while (subRule > 0 && ! ruleTok->isNT())
         {
            --subRule;
            ruleTok = rule->term(subRule);
         }

         // descend, unless we're in an endless loop
         if (ruleTok->isNT() && (ruleTok != current || subRule != 0) )
         {
            TRACE1( "Parser::parserLoop -- Descending into current rule %s from pos %d", ruleTok->name().c_ize(), subRule );
            pushParseFrame(static_cast<const NonTerminal*>(ruleTok), subRule + currentFrame->m_nStackDepth );
         }
         else
         {
            currentFrame->m_hypotesis++;

            while(currentFrame->m_hypotesis == currentFrame->m_owningToken->arity())
            {
               if( currentFrame->m_bErrorMode )
               {
                  TRACE1( "Parser::parserLoop -- Discarding unrecognized token \"%s\"in error mode", _p->m_tokenStack->back()->token().name().c_ize() );
                  trim(1);
                  // try the whole rule again.
                  currentFrame->m_hypotesis = 0;
                  break;
               }

               TRACE1( "Parser::parserLoop -- NonTerminal %s failed at %d, popping the stack",
                        currentFrame->m_owningToken->name().c_ize(), currentFrame->m_hypToken );

               if( &_p->m_pframes->front() == &_p->m_pframes->back() )
               {
                  MESSAGE2( "Parser::parserLoop -- Declaring failure.");
                  parseError();
                  break;
               }

               _p->m_pframes->pop_back();
               currentFrame = &_p->m_pframes->back();
               currentFrame->m_hypotesis++;
            }

            currentFrame->m_hypToken = 0;
         }

      }
   }

   MESSAGE( "Parser::parserLoop -- done" );
}


void Parser::setErrorMode( const Token* limitToken )
{
   _p->m_pframes->back().m_bErrorMode = true;
   _p->m_pframes->back().m_limitToken = limitToken;
}

void Parser::applyCurrentRule()
{
   fassert( ! _p->m_pframes->empty() );
   Private::ParseFrame& pf = _p->m_pframes->back();
   NonTerminal* rule = static_cast<NonTerminal*>(pf.m_owningToken->term(pf.m_hypotesis));
   TRACE("Parser::applyCurrentRule -- applying %s to stack %s",
            rule->name().c_ize(), dumpStack().c_ize() );
   if( rule->applyHandler() != 0 )
   {
      rule->applyHandler()(*this,*rule);
   }
   else
   {
      TRACE2("Parser::applyCurrentRule -- Synthezising a token \"%s\" simplifying %d",
               pf.m_owningToken->name().c_ize(), rule->arity()  );
      if( rule->arity() > 0 )
      {
         TokenInstance* current = getNextToken();
         TokenInstance* ti =  TokenInstance::alloc(current->line(), current->chr(), *pf.m_owningToken );
         simplify(rule->arity(), ti);
      }
      else
      {
         TokenInstance* ti =  TokenInstance::alloc(0, 0, *pf.m_owningToken );
         simplify(0, ti);
      }
   }

   // Reset all the frames frame and recheck current rule.
   popParseFrame();

   // clear the error frames if they'rebelow our frame.
   if ( _p->m_pframes->size() < _p->m_pErrorFrames->size() )
   {
      TRACE1("Parser::applyCurrentRule -- clearing error frame of lower rules (%d on %d)",
               _p->m_pframes->size(), _p->m_pErrorFrames->size());

      _p->m_pErrorFrames->clear();
   }

   //Private::FrameStack::iterator iter = _p->m_pframes->begin();
   //while( iter != _p->m_pframes->end() )
   {
      _p->m_pframes->back().m_hypotesis = 0;
      _p->m_pframes->back().m_hypToken = 0;
      _p->m_pframes->back().m_prio = 0;
      //++iter;
   }

   TRACE2("Parser::applyCurrentRule -- After applying %s stack %s",
            rule->name().c_ize(), dumpStack().c_ize() );
}


bool Parser::readNextToken()
{
   MESSAGE( "Parser::readNextToken -- Getting a new token." );

   Lexer* lexer = _p->m_lLexers.back();
   // we're done ?
   if( lexer == 0 )
   {
     MESSAGE2( "Parser::readNextToken -- done on lexer pop" );
     return false;
   }

   if ( m_bEOLGiven && m_bInteractive )
   {
      MESSAGE2( "Parser::readNextToken -- done on explicit user RETURN in interactive mode" );
      return false;
   }

   TokenInstance* ti = lexer->nextToken();
   // consume the tokens if requested.
   while( ti != 0 && m_consumeToken != 0 )
   {
      ti = lexer->nextToken();
      TRACE2( "Parser::readNextToken -- Consuming token \"%s\" up to \"%s\"",
               ti->token().name().c_ize(), m_consumeToken->name().c_ize() );
      if( &ti->token() == m_consumeToken )
      {
         // read one more.
         ti = lexer->nextToken();
         m_consumeToken = 0;
      }
   }

   while( ti == 0 )
   {
      if( m_bInteractive )
      {
         MESSAGE2( "Parser::readNextToken -- done on interactive lexer token shortage" );
         return false;
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
      ti = TokenInstance::alloc(0, 0, T_EOF );
   }

   if( (&ti->token() == &T_EOL) && ti->line() >= 0  )
   {
      // with this, we can return on next token request in interactive mode.
      m_bEOLGiven = true;
   }

   if( _p->m_pframes->back().m_limitToken == &ti->token() )
   {
      TRACE1( "Parser::parserLoop -- Exiting error mode because found limit token \"%s\"", ti->token().name().c_ize() )
      _p->m_pframes->back().m_limitToken = 0;
      _p->m_pframes->back().m_bErrorMode = false;
   }

   _p->m_tokenStack->push_back(ti);

   return true;
}

}
}
/* end of parser/parser.cpp */

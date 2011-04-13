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

#include <falcon/parser/parser.h>
#include <falcon/parser/lexer.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/state.h>
#include <falcon/parser/teof.h>
#include <falcon/codeerror.h>
#include <falcon/trace.h>

#include "./parser_private.h"

namespace Falcon {
namespace Parsing {

Parser::Private::Private():
   m_stackPos(0),
   m_nextTokenPos(0),
   m_nextToken(0)
{
}

Parser::Private::~Private()
{
   delete m_nextToken;
   clearTokens();
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
   TokenStack::iterator iter = m_vTokens.begin();
   while( iter != m_vTokens.end() )
   {
      delete *iter;
      ++iter;
   }
   m_vTokens.clear();
}


/** Resets the temporary values set in  top-level match. */
void Parser::Private::resetMatch()
{
   m_stackPos = 0;
}

//========================================================

Parser::Parser():
   m_ctx(0),
   m_bIsDone(false)
{
   _p = new Private;
}


Parser::~Parser()
{
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
   }
   else
   {
      throw new CodeError( ErrorParam( e_state, __LINE__, __FILE__ ).extra(name) );
   }
}


void Parser::popState()
{
   TRACE( "Parser::popState -- popping state", 0 );
   if ( _p->m_lStates.empty() )
   {
      throw new CodeError( ErrorParam( e_underflow, __LINE__, __FILE__ ).extra("Parser::popState") );
   }

   _p->m_lStates.pop_back();
   TRACE1( "Parser::popState -- now topmost state is '%s'", _p->m_lStates.back()->name().c_ize() );
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

   // If we have no error we succeeded.
   return _p->m_lErrors.empty();
}

void Parser::enumerateErrors( Parser::errorEnumerator& enumerator )
{
   Private::ErrorList::iterator iter = _p->m_lErrors.begin();
   while( iter != _p->m_lErrors.end() )
   {
      enumerator( *iter, ++iter == _p->m_lErrors.end() );
   }
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
   _p->m_lErrors.push_back(ErrorDef(code, uri, l, c, ctx, extra));
}


void Parser::addError( int code, const String& uri, int l, int c, int ctx  )
{
   _p->m_lErrors.push_back(ErrorDef(code, uri, l, c, ctx));
}


int32 Parser::tokenCount()
{
   return _p->m_vTokens.size();
}

int32 Parser::availTokens()
{
   return _p->m_vTokens.size() - _p->m_nextTokenPos;
}

TokenInstance* Parser::getNextToken()
{
   if( _p->m_nextTokenPos >= _p->m_vTokens.size() )
   {
      return 0;
   }

   return _p->m_vTokens[_p->m_nextTokenPos++];
}

void Parser::resetNextToken()
{
   _p->m_nextTokenPos = _p->m_stackPos;
}

void Parser::simplify( int32 tcount, TokenInstance* newtoken )
{
   TRACE( "Parser::simplify -- %d tokens -> %s",
         tcount, newtoken ? newtoken->token().name().c_ize() : "<nothing>" );

   if( tcount < 0 || tcount + _p->m_stackPos > _p->m_vTokens.size() )
   {
      throw new CodeError(ErrorParam(e_underflow, __LINE__, __FILE__ )
            .extra("Parser::simplify - tcount out of range"));
   }

   if( tcount != 0 )
   {
      size_t end = _p->m_stackPos + tcount;
      for( size_t pos = _p->m_stackPos; pos < end; ++pos )
      {
         delete _p->m_vTokens[pos];
      }

      _p->m_vTokens.erase( _p->m_vTokens.begin() + _p->m_stackPos, _p->m_vTokens.begin() + end );
   }

   if( newtoken != 0 )
   {
      _p->m_vTokens.insert( _p->m_vTokens.begin() + _p->m_stackPos, newtoken );
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
         popLexer();
         if( _p->m_lLexers.empty() )
         {
            lexer = 0;
            break;
         }
         else
         {
            lexer = _p->m_lLexers.back();
            TokenInstance* ti = lexer->nextToken();
         }
      }

      if( ti == 0 )
      {
         TRACE( "Parser::parserLoop -- Last loop with EOF as next", 0 );
         ti = new TokenInstance(0, 0, t_eof() );
      }

      _p->m_nextToken = ti;
      TRACE( "Parser::parserLoop -- read token '%s'", _p->m_nextToken->token().name().c_ize() );

      State* curState = _p->m_lStates.back();
      curState->process( *this );
      _p->m_vTokens.push_back( _p->m_nextToken );
      _p->m_nextToken = 0;
   }

   TRACE( "Parser::parserLoop -- done on request", 0 );
}


}
}

/* end of parser/parser.cpp */

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
#include <falcon/codeerror.h>

#include <deque>
#include <map>


namespace Falcon {
namespace Parser {

class Parser::Private
{
   friend class Parser;

   typedef std::deque<Lexer*> LexerStack;
   typedef std::deque<State*> StateStack;
   typedef std::deque<TokenInstance*> TokenStack;

   typedef std::deque<Parser::ErrorDef> ErrorList;

   typedef std::map<String, State*> StateMap;


   StateStack m_lStates;
   TokenStack m_lTokens;
   LexerStack m_lLexers;

   ErrorList m_lErrors;

   TokenInstance* m_nextToken;
   StateMap m_states;

   Private():
      m_nextToken(0)
   {
   }

   ~Private()
   {
      delete m_nextToken;

      {
         TokenStack::iterator iter = m_lTokens.begin();
         while( iter != m_lTokens.end() )
         {
            delete *iter;
            ++iter;
         }
      }

      {
         LexerStack::iterator iter = m_lLexers.begin();
         while( iter != m_lLexers.end() )
         {
            delete *iter;
            ++iter;
         }
      }
   }

};

//========================================================

Parser::Parser():
   m_ctx(0)
{
   _p = new Private;
}


Parser::~Parser()
{
   delete _p;
}


void Parser::addState( State& state )
{
   _p->m_states[state.name()] = &state;
}


void Parser::pushState( const String& name )
{
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
   if ( _p->m_lStates.empty() )
   {
      throw new CodeError( ErrorParam( e_underflow, __LINE__, __FILE__ ).extra("Parser::popState") );
   }

   _p->m_lStates.pop_back();
}


bool Parser::parse( const String& mainState )
{
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

   // Ok, we can start.
   Lexer* lexer = _p->m_lLexers.back();

   //....

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
 
}
}

/* end of parser/parser.cpp */
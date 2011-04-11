/*
   FALCON - The Falcon Programming Language.
   FILE: parser/rule.cpp

   Token for the parser subsystem.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 08 Apr 2011 21:17:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/rule.h>
#include <falcon/parser/parser.h>
#include "./parser_private.h"
#include <falcon/parser/token.h>
#include <falcon/parser/tokeninstance.h>
#include <falcon/parser/nonterminal.h>
#include <falcon/string.h>

#include <falcon/trace.h>

#include <vector>


namespace Falcon {
namespace Parsing {

class Rule::Private
{
private:
   friend class Rule;
   friend class Rule::Maker;

   typedef std::vector<Token*> TokenVector;
   TokenVector m_vTokens;
   
};

Rule::Maker::Maker( const String& name, Apply app ):
   m_name(name),
   m_apply(app)
{
   _p = new Rule::Private;
}

Rule::Maker::~Maker()
{
   delete _p;
}

Rule::Maker& Rule::Maker::t( Token& t )
{
   _p->m_vTokens.push_back( &t );
   return *this;
}


Rule::Rule( const String& name, Apply app )
{
   _p = new Rule::Private;
}


Rule::Rule( const Maker& m ):
   m_name( m.m_name ),
   m_apply( m.m_apply )
{
   _p = m._p;
   m._p = 0;
}


Rule::~Rule()
{
   delete _p;
}

Rule& Rule::t( Token& t )
{
   _p->m_vTokens.push_back( &t );
   return *this;
}


void Rule::apply( Parser& parser ) const
{
   parser.resetNextToken();
   m_apply(*this, parser);
}


t_matchType Rule::match( Parser& parser ) const
{
   TRACE( "Rule::match: %s", m_name.c_ize() );

   Parser::Private* pp = parser._p;
   size_t ppos = pp->m_stackPos;
   size_t ppos_end = pp->m_vTokens.size();

   Private::TokenVector::iterator riter = _p->m_vTokens.begin();
   Private::TokenVector::iterator riter_end = _p->m_vTokens.end();

   // check if there is a match by scanning the current token stack in the parser
   // -- and matching it against our tokens.
   while ( riter != riter_end && ppos < ppos_end )
   {
      Token* curTok = *riter;
      if (curTok->id() != pp->m_vTokens[ppos]->token().id() )
      {
         // is this a non-terminal token that we may simplify?
         if( curTok->isNT() && curTok->id() != m_parent->id() )
         {
            TRACE1( "Rule::match: %s -- at stack %d descending into %s",
               m_name.c_ize(), ppos, curTok->name().c_ize() );

            // push a new stack base for our checks.
            size_t piter_old = pp->m_stackPos;
            pp->m_stackPos = ppos;
            // perform a new check
            t_matchType mt = static_cast<NonTerminal*>(curTok)->match(parser);
            // restore old stack base
            pp->m_stackPos = piter_old;

            if( mt != t_match )
            {
               TRACE( "Rule::match: %s at step %d -- return on nonterminal '%s' %s",
                     m_name.c_ize(), ppos, curTok->name().c_ize(), mt == t_nomatch ? "failed" : "needs more tokens" );

               // the nonterminal token didn't match -- we can't proceed
               return mt;
            }
            
            // otherwise the token stack has been reduced to what we expected,
            // so we can continue.
            ppos_end = pp->m_vTokens.size();  // update token stack size.
         }
         else
         {
            TRACE( "Rule::match: %s at step %d -- failed", m_name.c_ize(), ppos );
            // terminal token mismatch -- we failed
            return t_nomatch;
         }
      }

      ++ppos;
      ++riter;
   }

   // do we have a perfect match?
   if( riter == riter_end )
   {
      TRACE( "Rule::match: %s -- success", m_name.c_ize(), ppos );
      return t_match;
   }
   else
   {
      // check the look-ahead token
      if( pp->m_nextToken->token().id() == (*riter)->id() )
      {
         TRACE( "Rule::match: %s -- too short", m_name.c_ize(), ppos );
         // if it matches, then we're too short
         return t_tooShort;
      }
   }

   TRACE( "Rule::match: %s -- final failure", m_name.c_ize(), ppos );
   // if any match fails...
   return t_nomatch;
}


}
}

/* end of parser/rule.cpp */

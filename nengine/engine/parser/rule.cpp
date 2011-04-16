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

   typedef std::vector<Token*> TokenVector;
   TokenVector m_vTokens;
   
};

//=================================================================
//

Rule::Rule()
{
   _p = new Rule::Private;
}

Rule::Rule( const String& name, Apply app ):
   m_name(name),
   m_apply(app)
{
   _p = new Rule::Private;
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

try_again:
   TRACE( "Rule::match(%s)", m_name.c_ize() );

   Parser::Private* pp = parser._p;
   size_t ppos = pp->m_stackPos;
   size_t ppos_end = pp->m_vTokens.size();

   Private::TokenVector::iterator riter = _p->m_vTokens.begin();
   Private::TokenVector::iterator riter_end = _p->m_vTokens.end();

   t_matchType checkStatus = t_nomatch;

   size_t leftmost_right_assoc = (size_t)-1;
   int highPrio = 0;

   // check if there is a match by scanning the current token stack in the parser
   // -- and matching it against our tokens.
   while ( riter != riter_end && ppos < ppos_end )
   {
      Token* curTok = *riter;
      const Token* stackToken = &pp->m_vTokens[ppos]->token();
      
      // is this a non-terminal token that we may simplify?
      // TODO: we can know if we have to descend AGAIN if the previous loop was too short.
      // ---- but atm we're discardign this information.
      if( curTok->isNT() && ( ppos > pp->m_stackPos || curTok->id() != m_parent->id() ) )
      {
         TRACE1( "Rule::match(%s) -- at stack %d descending into %s",
            m_name.c_ize(), ppos, curTok->name().c_ize() );

         // push a new stack base for our checks.
         size_t piter_old = pp->m_stackPos;
         pp->m_stackPos = ppos;
         // perform a new check
         t_matchType mt = static_cast<NonTerminal*>(curTok)->match(parser);
         // restore old stack base
         pp->m_stackPos = piter_old;

         if( mt == t_match )
         {
            TRACE( "Rule::match(%s) at step %d -- matched sub-rule '%s'",
                  m_name.c_ize(), ppos, curTok->name().c_ize() );
            TRACE1( "  -- stack now: %s ", parser.dumpStack().c_ize() );
            
            // Try again with the same position till this rule matches.
            ppos_end = pp->m_vTokens.size();  // update token stack size.
            checkStatus = t_match;
            continue;
         }
         else if( mt == t_tooShort )
         {
            // Too short is returned only if the rule arrives to check the next token.
            if( curTok->id() != m_parent->id() || pp->m_nextToken->token().isRightAssoc() )
            {
               // we're too short as well, because either we have a sub-rule that is not matching,
               // or we're a right-associativity rule.
               TRACE( "Rule::match: %s at step %d -- return too short sub-rule '%s'",
                  m_name.c_ize(), ppos, curTok->name().c_ize() );

               return t_tooShort;
            }

            // curtok is parent; that is, we're checking a recursive rule, which
            // has been deemed too-short. 
            
         }
         // Otherwise the rule failed or was considered incompete.
         // In either case, we must proceed.
      }

      TRACE1( "Rule::match(%s) -- at stack %d matching %s with %s", m_name.c_ize(), ppos,
            curTok->name().c_ize(), pp->m_vTokens[ppos]->token().name().c_ize() );

      if (curTok->id() != stackToken->id() )
      {         
         TRACE( "Rule::match(%s) at stack %d -- failed( %s vs %s )", 
               m_name.c_ize(), ppos, curTok->name().c_ize(), stackToken->name().c_ize() );
         // terminal token mismatch -- we failed
         return t_nomatch;
      }

      // Prepare in case of right associativity
      if (curTok->id() == pp->m_nextToken->token().id() )
      {
         leftmost_right_assoc = ppos;
      }
      else if ( curTok->prio() != 0 &&  (curTok->prio() < highPrio || highPrio == 0) )
      {
         highPrio = curTok->prio();
      }
      
      ++ppos;
      ++riter;
   }

   // there's more stack than tokens in the rule?
   if( ppos < ppos_end )
   {
      // if the next token has higher priority...
      if (pp->m_nextToken->token().prio() != 0 && highPrio > pp->m_nextToken->token().prio())
      {
         //... let the rule to try again.
         return t_tooShort;
      }

      // but... is the next token right-associative and found in the rule?
      if( leftmost_right_assoc != (size_t)-1 && pp->m_vTokens[ppos]->token().isRightAssoc() )
      {
         // then descend in ourselves, and see if the rule is incomplete.
         TRACE( "Rule::match(%s) -- too short, descending %d on right assoc %s",
            m_name.c_ize(), leftmost_right_assoc, pp->m_nextToken->token().name().c_ize() );

         // push a new stack base for our checks.
         size_t piter_old = pp->m_stackPos;
         pp->m_stackPos = leftmost_right_assoc+1;
         // perform a new check on ourselves.
         t_matchType mt = match( parser );
         // on match, apply and try ourselves again
         if( mt == t_match )
         {
            TRACE( "Rule::match(%s) -- Right associativity matched on short rule. Simplifying and trying again",
               m_name.c_ize() );
            apply(parser);
            pp->m_stackPos = piter_old;
            goto try_again;
         }
         // restore old stack base
         pp->m_stackPos = piter_old;

         // if the rule didn't match, we're ok to fail
      }

      TRACE( "Rule::match(%s) -- failure -- stack not completely matched (%d vs %d)", m_name.c_ize(),
         ppos, ppos_end );

      return t_nomatch;
   }

   // do we have a perfect match?
   if( riter == riter_end  )
   {
      // if the next token has higher priority...
      if (pp->m_nextToken->token().prio() != 0 && highPrio > pp->m_nextToken->token().prio())
      {
         //... let the rule to try again.
         return t_tooShort;
      }

      // but... is the next token right-associative and found in the rule?
      if( leftmost_right_assoc != (size_t)-1 && pp->m_nextToken->token().isRightAssoc() )
      {
         // then descend in ourselves, and see if the rule is incomplete.
         TRACE( "Rule::match(%s) -- descending at %d to check for right associativity on token %s",
            m_name.c_ize(), leftmost_right_assoc, pp->m_nextToken->token().name().c_ize() );

         // push a new stack base for our checks.
         size_t piter_old = pp->m_stackPos;
         pp->m_stackPos = leftmost_right_assoc+1;
         // perform a new check on ourselves.
         t_matchType mt = match( parser );
         // on match, apply and try ourselves again
         if( mt == t_match )
         {
            TRACE( "Rule::match(%s) -- Right associativity matched. Simplifying and trying again",
               m_name.c_ize() );
            apply(parser);
            pp->m_stackPos = piter_old;
            goto try_again;
         }
         // restore old stack base
         pp->m_stackPos = piter_old;

         // if the rule was too short, then we're too short.
         if( mt == t_tooShort )
         {
            return t_tooShort;
         }

         // if the rule didn't match, we're ok to match.
      }

      TRACE( "Rule::match(%s) -- success", m_name.c_ize(), ppos );
      return t_match;
   }

   checkStatus = (*riter)->id() == pp->m_nextToken->token().id() ? t_tooShort : t_nomatch;
   
   TRACE( "Rule::match(%s) -- exausted rule at %d (%s)", m_name.c_ize(), ppos,
      checkStatus == t_tooShort ? "too short" : "no match" );
   // if any match fails...
   return checkStatus;
}


}
}

/* end of parser/rule.cpp */

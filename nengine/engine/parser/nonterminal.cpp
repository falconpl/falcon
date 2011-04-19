/*
   FALCON - The Falcon Programming Language.
   FILE: parser/nonterminal.cpp

   Token representing a non-terminal grammar symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 12:54:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/nonterminal.h>
#include <falcon/parser/rule.h>
#include <falcon/trace.h>

#include <deque>


namespace Falcon {
namespace Parsing {

//==========================================================
// Helper classes
//

class NonTerminal::Private
{
   friend class NonTerminal;
   Private() {}
   ~Private() {}

   typedef std::deque<Rule*> RuleList;
   RuleList m_rules;
};

//=======================================================
// Main nonterminal class
//

NonTerminal::NonTerminal(const String& name,  bool bRightAssoc ):
   Token(name)
{
   m_eh = 0;
   m_bRightAssoc = bRightAssoc;
   m_bNonTerminal = true;
   _p = new Private;
}

NonTerminal::NonTerminal():
   Token("Unnamed NT")
{
   m_bNonTerminal = true;
   m_eh = 0;
   _p = new Private;
}


NonTerminal::~NonTerminal()
{
   delete _p;
}

NonTerminal& NonTerminal::r(Rule& rule)
{
   _p->m_rules.push_back( &rule );
   rule.parent(*this);
   return *this;
}


t_matchType NonTerminal::match( Parser& parser )
{
   TRACE( "NonTerminal::match %s -- scanning", name().c_ize() );

   // loop through our rules.
   Private::RuleList::iterator iter = _p->m_rules.begin();
   Private::RuleList::iterator end = _p->m_rules.end();
   Rule* winner = 0;

   while( iter != end )
   {
      Rule* rule = *iter;
      t_matchType ruleMatch = rule->match( parser );
      if( ruleMatch == t_match )
      {
         // wow, we have a winner.
         if (winner == 0 )
         {
            winner = rule;
            TRACE1( "NonTerminal::match %s -- electing winner %s",
                  name().c_ize(), winner->name().c_ize() );
         }
      }
      else if( ruleMatch == t_tooShort )
      {
         // the rule cannot be decided.
         TRACE( "NonTerminal::match %s -- return because non-decidible ", name().c_ize() );
         return t_tooShort;
      }

      // If it doesn't match, we don't care.
      ++iter;
   }

   // ok, do we have a winner?
   if( winner != 0 )
   {
      TRACE1( "NonTerminal::match %s -- Applying winner %s", name().c_ize(), winner->name().c_ize() );
      winner->apply( parser );
      return t_match;
   }

   TRACE( "NonTerminal::match %s -- return with no match", name().c_ize() );
   return t_nomatch;
}

}
}

/* end of parser/nonterminal.cpp */

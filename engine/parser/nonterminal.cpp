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
#define SRC "engine/parser/nonterminal.cpp"

#include <falcon/parser/nonterminal.h>
#include <falcon/parser/parser.h>
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
   Token(name),
   m_maxArity(0),
   m_bRecursive(false),
   m_bGreedy(false)
{
   m_eh = 0;
   m_bRightAssoc = bRightAssoc;
   m_bNonTerminal = true;
   _p = new Private;
}


NonTerminal::NonTerminal():
   Token("Unnamed NT"),
   m_maxArity(0),
   m_bRecursive(false),
   m_bGreedy(false)
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

   // after setting parentship, we'll know if the rule is recursive.
   rule.parent(*this);
   if( rule.isRecursive() )
   {
      m_bRecursive = true;
   }

   if ( m_maxArity < rule.arity() )
   {
      m_maxArity = rule.arity();
   }

   return *this;
}


bool NonTerminal::findPaths( Parser& p ) const
{
   TRACE1( "NonTerminal::findPaths -- scanning '%s'", name().c_ize() );

   // initialize frame status
   int nBaseFrames = p.frameDepth();
   int nBaseRules = p.rulesDepth();

   // loop through our rules.
   Private::RuleList::iterator iter = _p->m_rules.begin();
   Private::RuleList::iterator end = _p->m_rules.end();

   while( iter != end )
   {
      const Rule* rule = *iter;

      p.addRuleToPath(rule);
      if( rule->match( p, false ) )
      {
         TRACE1( "NonTerminal::findPaths(%s) -- match '%s'", name().c_ize(), rule->name().c_ize() );
         return true;
      }
      
      p.unroll( nBaseFrames, nBaseRules ); // discard also the old winning rule
   
      // If it doesn't match, we don't care.
      ++iter;
   }

   TRACE1( "NonTerminal::findPaths(%s) -- no match", name().c_ize() );
   return false;
}

void NonTerminal::addFirstRule( Parser& p ) const
{
   // loop through our rules.
   Private::RuleList::iterator iter = _p->m_rules.begin();

   TRACE1( "NonTerminal::addFirstRule -- adding '%s'", (*iter)->name().c_ize() );
   p.addRuleToPath((*iter));
}

}
}

/* end of parser/nonterminal.cpp */

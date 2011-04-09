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
#include <vector>

namespace Falcon {
namespace Parser {

class NonTerminal::Private
{
   friend class NonTerminal;
   friend class NonTerminal::Maker;
   Private() {}
   ~Private() {}

   std::vector<Rule*> m_rules;
};


NonTerminal::Maker::Maker(const String& name):
   m_name(name)
{
   _p = new NonTerminal::Private;
}


NonTerminal::Maker::~Maker()
{
   delete _p;
}

NonTerminal::Maker& NonTerminal::Maker::r(Rule& rule)
{
   _p->m_rules.push_back(&rule);
   return *this;
}


NonTerminal::NonTerminal(const String& name):
   Token(name)
{
   _p = new Private;
}

NonTerminal::NonTerminal(const Maker& maker ):
   Token(maker.m_name)
{
   _p = maker._p;
   maker._p = 0;
}

NonTerminal::~NonTerminal()
{
   delete _p;
}

NonTerminal& NonTerminal::r(Rule& rule)
{
   _p->m_rules.push_back( &rule );
   return *this;
}

}
}

/* end of parser/nonterminal.cpp */

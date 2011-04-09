/*
   FALCON - The Falcon Programming Language.
   FILE: parser/state.cpp

   Token representing a non-terminal grammar symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 09 Apr 2011 17:32:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/parser/state.h>
#include <falcon/parser/nonterminal.h>
#include <vector>

namespace Falcon {
namespace Parser {

class State::Private
{
   friend class State;
   friend class State::Maker;
   Private() {}
   ~Private() {}

   std::vector<NonTerminal*> m_nt;
};


State::Maker::Maker(const String& name):
   m_name(name)
{
   _p = new State::Private;
}


State::Maker::~Maker()
{
   delete _p;
}

State::Maker& State::Maker::n(NonTerminal& nt)
{
   _p->m_nt.push_back(&nt);
   return *this;
}


State::State(const String& name):
   m_name(name)
{
   _p = new Private;
}

State::State(const State::Maker& maker ):
   m_name(maker.m_name)
{
   _p = maker._p;
   maker._p = 0;
}

State::~State()
{
   delete _p;
}

State& State::n(NonTerminal& nt)
{
   _p->m_nt.push_back( &nt );
   return *this;
}

}
}

/* end of parser/state.cpp */


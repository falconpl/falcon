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
#include <falcon/trace.h>
#include <vector>

#include "falcon/parser/parser.h"

namespace Falcon {
namespace Parsing {

class State::Private
{
   friend class State;
   Private() {}
   ~Private() {}

   typedef std::vector<NonTerminal*> NTList;
   NTList m_nt;
};

State::State():
   m_name("Unnamed State")
{
   _p = new Private;
}

State::State(const String& name):
   m_name(name)
{
   _p = new Private;
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


void State::process( Parser& parser )
{
   TRACE("State::process -- enter %s", name().c_ize() );

   // Process all the rules in a state
   Private::NTList::iterator iter = _p->m_nt.begin();

   bool bTryAgain;

   while( iter != _p->m_nt.end() )
   {
      NonTerminal* nt = *iter;

      TRACE1("State::process -- checking %s", nt->name().c_ize() );
      
      t_matchType mt = nt->match( parser );

      TRACE1("State::process -- nt-token %s %smatch",
               nt->name().c_ize(), mt == t_match ? "": "doesn't " );

      if ( mt == t_match )
      {
         return;
      }

      if( mt == t_tooShort )
      {
         bTryAgain = true;
      }
      
      ++iter;
   }

   TRACE1("State::process -- exit without match %s", name().c_ize() );

   if( ! bTryAgain && parser.availTokens() > 2 )
   {
      parser.syntaxError();
   }
}

}
}

/* end of parser/state.cpp */


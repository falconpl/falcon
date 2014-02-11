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

#define SRC "engine/parser/state.cpp"

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


bool State::findPaths( Parser& parser )
{
   TRACE1("State::findPaths -- enter %s", name().c_ize() );

   // Process all the non terminals in this state
   Private::NTList::iterator iter = _p->m_nt.begin();
   while( iter != _p->m_nt.end() )
   {
      NonTerminal* nt = *iter;
      //TRACE2("State::findPaths -- checking %s", nt->name().c_ize() );

      parser.pushParseFrame(nt,0);

      // don't allow ambiguity
      /*if ( nt->findPaths( parser ) )
      {
         TRACE2("State::findPaths -- nt-token %s match",
               nt->name().c_ize() );
         return true;
      }
      */
      
      
      ++iter;
   }

   TRACE1("State::findPaths -- exit without match %s", name().c_ize() );
   return false;
}

}
}

/* end of parser/state.cpp */


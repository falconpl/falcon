/*
   FALCON - The Falcon Programming Language.
   FILE: syntree.cpp

   Syntactic tree item definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Jan 2011 20:37:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/syntree.h>
#include <falcon/vm.h>
#include <falcon/codeframe.h>

namespace Falcon
{

SynTree::SynTree()
{}

SynTree::~SynTree()
{
   for( int i = 0; i < m_steps.size(); ++i )
   {
      delete m_steps[i];
   }
}


void SynTree::toString( String& tgt ) const
{
   for( int i = 0; i < m_steps.size(); ++i )
   {
      tgt += m_steps[i]->toString() + "\n";
   }
}


void SynTree::perform( VMachine* vm ) const
{
   // perform action is to register ourselves as sequence to be parsed.
   vm->pushCode( this );
}


void SynTree::apply( VMachine* vm ) const
{
   // get the current step.
   CodeFrame& cf = vm->currentCode();
   if (cf.m_seqId >= m_steps.size() )
   {
      // we're done.
      vm->popCode();
      return;
   }

   PStep* step = m_steps[ cf.m_seqId++ ];
   step->perform(vm);

   //TODO: continue to perform if not in debug and pstep was simple.
}

}

/* end of syntree.cpp */

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
#include <falcon/statement.h>

namespace Falcon
{

SynTree::SynTree()
{
   apply = apply_;
}

SynTree::~SynTree()
{
   for( size_t i = 0; i < m_steps.size(); ++i )
   {
      delete m_steps[i];
   }
}


void SynTree::describe( String& tgt ) const
{
   for( size_t i = 0; i < m_steps.size(); ++i )
   {
      tgt += m_steps[i]->describe() + "\n";
   }
}

void SynTree::apply_( const PStep* ps, VMContext* ctx )
{
   const SynTree* self = static_cast<const SynTree*>(ps);

   // get the current step.
   CodeFrame& cf = ctx->currentCode();
   if (cf.m_seqId >= (int) self->m_steps.size() )
   {
      // we're done.
      ctx->popCode();
      return;
   }

   Statement* step = self->m_steps[ cf.m_seqId++ ];
   step->prepare(ctx);
}


void SynTree::set( int pos, Statement* p )  {
   delete m_steps[pos];
   m_steps[pos] = p;
}

void SynTree::remove( int pos )
{
     Statement* p = m_steps[ pos ];
     m_steps.erase( m_steps.begin()+pos );
     delete p;
}
}

/* end of syntree.cpp */

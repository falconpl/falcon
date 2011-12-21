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

#include <falcon/trace.h>
#include <falcon/syntree.h>
#include <falcon/vm.h>
#include <falcon/codeframe.h>
#include <falcon/statement.h>
#include <falcon/symboltable.h>

#include <vector>

namespace Falcon
{

class SynTree::Private
{
public:
   typedef std::vector<Statement*> Steps;
   Steps m_steps;

   Private() {}

   ~Private()
   {
      Steps::iterator iter = m_steps.begin();
      while( iter != m_steps.end() )
      {
         delete *iter;
         ++iter;
      }
   }
};


SynTree::SynTree():
   _p( new Private ),
   m_locals(0),
   m_head(0)
{
   /** Mark this as a composed class */
   m_bIsComposed = true;
   apply = apply_empty_;
}


SynTree::~SynTree()
{
   delete _p;
   delete m_locals;
}


void SynTree::describeTo( String& tgt ) const
{
   for( size_t i = 0; i < _p->m_steps.size(); ++i )
   {
      tgt += _p->m_steps[i]->describe() + "\n";
   }
}


SymbolTable* SynTree::locals( bool bmake )
{
   if( m_locals == 0 && bmake )
   {
      m_locals = new SymbolTable();
   }
   
   return m_locals;
}


void SynTree::apply_( const PStep* ps, VMContext* ctx )
{
   const SynTree* self = static_cast<const SynTree*>(ps);
   // get the current step.
   CodeFrame& cf = ctx->currentCode();
   
   register Private::Steps& steps = self->_p->m_steps;
   length_t len = steps.size();
   int& seqId = cf.m_seqId;
   
   TRACE1( "Syntree::apply -- %p with %d/%d", ps, seqId, len );
   
   while( seqId < (int) len )
   {
      Statement* step = steps[ seqId++ ];
      if( ctx->stepInYield(step, cf) )
      {
         TRACE2( "Syntree::apply -- going deep at step %d \"%s\"", 
                     seqId-1, 
                     step->oneLiner().c_ize() );
         return;
      }
   }

   TRACE2( "Syntree::apply -- preparing \"%s\"", step->oneLiner().c_ize() );
   ctx->popCode();
}


void SynTree::apply_empty_( const PStep*, VMContext* ctx )
{
   // we don't exist -- and should not have been generated, actually
   ctx->popCode();
}


void SynTree::apply_single_( const PStep* ps, VMContext* ctx )
{
   const SynTree* self = static_cast<const SynTree*>(ps);
   register const PStep* singps = self->m_single;
   ctx->resetCode( singps );
   self->m_single->apply(singps, ctx);
}


void SynTree::set( int pos, Statement* p )  {
   delete _p->m_steps[pos];
  _p->m_steps[pos] = p;
}


void SynTree::remove( int pos )
{
  Statement* p =_p->m_steps[ pos ];
  _p->m_steps.erase( _p->m_steps.begin()+pos );
  delete p;
}


void SynTree::insert( int pos, Statement* step )
{
   _p->m_steps.insert( _p->m_steps.begin()+pos, step );
   size_t size = _p->m_steps.size();
   if( size == 2 && apply == apply_single_ )
   {
      apply = apply_;
   }
   else if( size == 1 && apply == apply_empty_ )
   {
      m_single = step;
      apply = apply_single_;
   }
}


SynTree& SynTree::append( Statement* step )
{
   _p->m_steps.push_back( step );
   size_t size = _p->m_steps.size();
   
   // check also the previous function to prevent overwriting.
   if( size == 2 && apply == apply_single_ )
   {
      apply = apply_;
   }
   else if( size == 1 && apply == apply_empty_ )
   {
      m_single = step;
      apply = apply_single_;
   }
   
   return *this;
}


int SynTree::size() const
{
   return _p->m_steps.size();
}


bool SynTree::empty() const
{
   return _p->m_steps.empty();
}


Statement* SynTree::first() const
{
   return _p->m_steps.front();
}


Statement* SynTree::last() const
{
   return _p->m_steps.back();
}


Statement* SynTree::at( int pos ) const
{ 
   return _p->m_steps[pos];
}

}

/* end of syntree.cpp */

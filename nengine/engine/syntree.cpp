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
   m_locals(0)
{
   apply = apply_;
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
   TRACE1( "Syntree::apply -- %p with %d/%d", ps, cf.m_seqId, self->_p->m_steps.size() );
   
   register Private::Steps& steps = self->_p->m_steps;
   register length_t len = steps.size();
   register Statement* step = steps[ cf.m_seqId++ ];
      
   if (cf.m_seqId >= (int) len )
   {
      // we're done.
      ctx->popCode();
      if( cf.m_seqId > (int) len )
      {
         MESSAGE2( "Syntree::exiting now" );
         // yeah, done for good 
         // -- step is invalid, the sequence was over before we got here.
         // -- This should happen only when the syntree was empty -- which
         // -- is pathological, as empty syntrees should have been purged
         return;
      }
      else
      {
         MESSAGE2( "Syntree::exiting next step" );         
      }
   }

   TRACE2( "Syntree::apply -- preparing \"%s\"", step->oneLiner().c_ize() );
   step->prepare(ctx);
}



void SynTree::apply_single_( const PStep* ps, VMContext* ctx )
{
   const SynTree* self = static_cast<const SynTree*>(ps);
   ctx->popCode();
   self->m_single->prepare(ctx);
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
   if( size == 2 )
   {
      apply = apply_;
   }
   else if( size == 1 )
   {
      m_single = step;
      apply = apply_single_;
   }
}


SynTree& SynTree::append( Statement* step )
{
   _p->m_steps.push_back( step );
   size_t size = _p->m_steps.size();
   
   if( size == 2 )
   {
      apply = apply_;
   }
   else if( size == 1 )
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

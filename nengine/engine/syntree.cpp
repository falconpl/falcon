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
   if (cf.m_seqId >= (int) self->_p->m_steps.size() )
   {
      // we're done.
      ctx->popCode();
      return;
   }

   Statement* step = self->_p->m_steps[ cf.m_seqId++ ];
   step->prepare(ctx);
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
}


SynTree& SynTree::append( Statement* step )
{
   _p->m_steps.push_back( step );
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

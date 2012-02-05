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
#include <falcon/expression.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>

#include "psteps/exprvector_private.h"

namespace Falcon
{

class SynTree::Private
{
public:
   typedef TSVector_Private<Statement> Steps;
   Steps m_steps;

   Private() {}
   Private( const Private& other, SynTree* owner ):
      m_steps( other.m_steps, owner) 
   {}
   
   ~Private() {}
};


SynTree::SynTree( int line, int chr ):
   TreeStep( TreeStep::e_cat_syntree, line, chr ),
   _p( new Private ),
   m_locals(0),
   m_head(0),
   m_selector(0)
{
   static Class* syntoken = Engine::instance()->synclasses()->m_cls_st;
   m_class = syntoken;
   
   /** Mark this as a composed class */
   m_bIsComposed = true;
   apply = apply_empty_;
}


SynTree::SynTree( const SynTree& other ):
   TreeStep( TreeStep::e_cat_syntree ),
   _p( 0 ),
   m_locals(0),
   m_head(0),
   m_selector(0)
{   
   /** Mark this as a composed class */
   m_bIsComposed = true;  
   
   _p = new Private(*other._p, this);
   if( other.m_locals != 0 )
   {
      m_locals = new SymbolTable( *other.m_locals );
   }
   
   if( other.m_head != 0 ) {
      m_head = other.m_head;
   }
   
   if( other.m_selector != 0 )
   {
      m_selector = other.m_selector->clone();
      m_selector->setParent(this);
   }
   
   setApply();
}


SynTree::~SynTree()
{
   delete _p;
   delete m_locals;
   delete m_selector;
   // we don't own the head symbol
}


bool SynTree::selector( Expression* expr )
{
   if ( ! expr->setParent( this ) )
   {
      return false;
   }
   
   delete m_selector;
   m_selector = expr;
   return true;
}


void SynTree::describeTo( String& tgt, int depth ) const
{
   String prefix = String( " " ).replicate( depth * depthIndent );
   for( size_t i = 0; i < _p->m_steps.m_exprs.size(); ++i )
   {
      if ( i > 0 ) tgt += "\n";
      tgt += prefix + _p->m_steps.m_exprs[i]->describe( depth + 1 );
   }
}


void SynTree::oneLinerTo( String& tgt ) const
{
   // todo: represent the first one?
   tgt = "...";
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
   length_t len = steps.m_exprs.size();
   int& seqId = cf.m_seqId;
   
   TRACE1( "Syntree::apply -- %p with %d/%d", ps, seqId, len );
   
   while( seqId < (int) len )
   {
      Statement* step = steps.m_exprs[ seqId++ ];
      if( ctx->stepInYield(step, cf) )
      {
         TRACE2( "Syntree::apply -- going deep at step %d \"%s\"", 
                     seqId-1, 
                     step->oneLiner().c_ize() );
         return;
      }
   }

   TRACE2( "Syntree::apply -- preparing \"%s\"", self->oneLiner().c_ize() );
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
   register const PStep* singps = self->_p->m_steps.m_exprs[0];
   ctx->resetCode( singps );
   singps->apply(singps, ctx);
}


int SynTree::arity() const
{
   return _p->m_steps.arity();
}


TreeStep* SynTree::nth( int pos ) const
{
   return _p->m_steps.nth( pos );
}

bool SynTree::nth( int pos, TreeStep* step )
{
   if( step == 0 
      || step->category() != TreeStep::e_cat_statement 
      || ! step->setParent(this) ) 
   {
      return false;
   }
   
   return _p->m_steps.nth( pos, static_cast<Statement*>(step), this );
}

bool SynTree::insert( int pos, TreeStep* step )
{
   if( step == 0 
      || step->category() != TreeStep::e_cat_statement 
      || ! step->setParent(this)) 
   {
      return false;
   }
   
   if( ! _p->m_steps.insert( pos, static_cast<Statement*>(step), this ) ) return false;
   if ( _p->m_steps.arity() <= 2 ){
      setApply();   
   }
   return true;
}

bool SynTree::remove( int pos )
{
   if( ! _p->m_steps.remove( pos ) ) return false;
   
   if ( _p->m_steps.arity() <= 2 ){
      setApply();
   }
   return true;
}


SynTree& SynTree::append( Statement* step )
{
   if( ! step->setParent(this) ) return *this;
   _p->m_steps.m_exprs.push_back( step );
   if ( _p->m_steps.arity() <= 2 ){
      setApply();   
   }
   return *this;
}

size_t SynTree::size() const
{
   return _p->m_steps.m_exprs.size();
}


bool SynTree::empty() const
{
   return _p->m_steps.m_exprs.empty();
}


Statement* SynTree::first() const
{
   return _p->m_steps.m_exprs.front();
}


Statement* SynTree::last() const
{
   return _p->m_steps.m_exprs.back();
}


Statement* SynTree::at( int pos ) const
{ 
   return _p->m_steps.m_exprs[pos];
}

void SynTree::setApply()
{ 
   switch( _p->m_steps.m_exprs.size() )
   {
      case 0:
         apply = apply_empty_;
         break;
         
      case 1:
         apply = apply_single_;
         break;
         
      default:
         apply = apply_;
   }
}

}

/* end of syntree.cpp */

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

#undef SRC
#define SRC "engine/syntree.cpp"

#include <falcon/trace.h>
#include <falcon/syntree.h>
#include <falcon/vm.h>
#include <falcon/codeframe.h>
#include <falcon/statement.h>
#include <falcon/expression.h>
#include <falcon/syntree.h>
#include <falcon/symbol.h>
#include <falcon/textwriter.h>

#include <falcon/engine.h>
#include <falcon/synclasses.h>
#include <falcon/synclasses_id.h>

#include "psteps/exprvector_private.h"

namespace Falcon
{

class SynTree::Private
{
public:
   typedef TSVector_Private<TreeStep> Steps;
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
   m_head(0),
   m_selector(0)
{
   static Class* syntoken = Engine::instance()->synclasses()->m_cls_SynTree;
   m_handler = syntoken;
   
   /** Mark this as a composed class */
   m_bIsComposed = true;
   apply = apply_;
}


SynTree::SynTree( const SynTree& other ):
   TreeStep( other ),
   _p( 0 ),
   m_head(0),
   m_selector(0)
{   
   /** Mark this as a composed class */
   m_bIsComposed = true;  
   
   _p = new Private(*other._p, this);
   
   if( other.m_head != 0 ) {
      m_head = other.m_head;
   }
   
   if( other.m_selector != 0 )
   {
      m_selector = other.m_selector->clone();
      m_selector->setParent(this);
   }
   apply = apply_;
}


SynTree::~SynTree()
{
   delete _p;
   dispose( m_selector );
   if( m_head != 0 ) {
      m_head->decref();
   }
   // we don't own the head symbol
}


void SynTree::target( const Symbol* s )
{
   s->incref();
   if( m_head != 0 ) {
      m_head->decref();
   }
   m_head = s;
}


bool SynTree::selector( TreeStep* expr )
{
   if ( expr != 0 && ! expr->setParent( this ) )
   {
      return false;
   }
   
   dispose( m_selector );
   m_selector = expr;
   return true;
}


void SynTree::render( TextWriter* tw, int32 depth ) const
{
  for( size_t i = 0; i < _p->m_steps.m_exprs.size(); ++i )
  {
     _p->m_steps.m_exprs[i]->render( tw, depth );
  }
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
   
   // execute the first step without popping.
   if( seqId > 0 )
   {
      ctx->popData();
   }
   else if( len == 0 )
   {
      ctx->pushData(Item());
      ctx->popCode();
      return;
   }

   --len;
   while( seqId < (int32)len )
   {
      TreeStep* step = steps.m_exprs[ seqId++ ];
      if( ctx->stepInYield(step, cf) )
      {
         TRACE2( "Syntree::apply -- going deep at step %d \"%s\"", 
                     ctx->currentCode().m_seqId-1, 
                     step->describe().c_ize() );
         return;
      }


      ctx->popData();

      /*
      TreeStep* step = steps.m_exprs[ seqId++ ];
      ctx->pushCode(step);
      return;
      */
   }

   // step in the last and let it go.
   ctx->resetCode(steps.m_exprs[seqId]);
}


int SynTree::arity() const
{
   return _p->m_steps.arity();
}


TreeStep* SynTree::nth( int pos ) const
{
   return _p->m_steps.nth( pos );
}

bool SynTree::setNth( int pos, TreeStep* step )
{
   if( step == 0 || step->parent() != 0)
   {
      return false;
   }
   
   return _p->m_steps.nth( pos, step, this );
}

bool SynTree::insert( int pos, TreeStep* step )
{
   if( step == 0 || step->parent() != 0)
   {
      return false;
   }

   if( ! _p->m_steps.insert( pos, step, this ) )
   {
      return false;
   }
   return true;
}

bool SynTree::remove( int pos )
{
   if( ! _p->m_steps.remove( pos ) )
   {
      return false;
   }

   return true;
}


TreeStep* SynTree::detach( int pos )
{
   if( pos < 0 || pos >= (int) _p->m_steps.m_exprs.size() )
   {
      return 0;
   }
   TreeStep* ts = _p->m_steps.m_exprs[pos];
   ts->setParent(0);
   _p->m_steps.m_exprs.erase(_p->m_steps.m_exprs.begin() + pos);


   return ts;
}

void SynTree::clear()
{
   _p->m_steps.clear();
}

bool SynTree::append( TreeStep* step )
{
   if( step == 0 || ! step->setParent(this) )
   {
      return false;
   }
   _p->m_steps.m_exprs.push_back( step );

   return true;
}

size_t SynTree::size() const
{
   return _p->m_steps.m_exprs.size();
}


bool SynTree::empty() const
{
   return _p->m_steps.m_exprs.empty();
}


TreeStep* SynTree::first() const
{
   return _p->m_steps.m_exprs.front();
}


TreeStep* SynTree::last() const
{
   return _p->m_steps.m_exprs.back();
}


TreeStep* SynTree::at( int pos ) const
{ 
   return _p->m_steps.m_exprs[pos];
}


}

/* end of syntree.cpp */

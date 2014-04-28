/*
   FALCON - The Falcon Programming Language.
   FILE: expraccumulator.cpp

   Syntactic tree item definitions -- Case (switch branch selector)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 16 Mar 2013 14:00:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#undef SRC
#define SRC "engine/psteps/expraccumulator.cpp"

#include <falcon/psteps/expraccumulator.h>
#include <falcon/psteps/exprvalue.h>
#include <falcon/trace.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/itemarray.h>

#include <falcon/textwriter.h>

#include <falcon/stdsteps.h>
#include <falcon/synclasses.h>
#include <falcon/engine.h>

#include <vector>

namespace Falcon {


ExprAccumulator::ExprAccumulator( int line, int chr ):
   Expression( line, chr ),
   m_target(0),
   m_filter(0),
   m_vector(0),
   m_stepGenIter(this),
   m_stepGenNext(this),
   m_stepTakeNext(this),
   m_stepAfterFilter(this),
   m_stepAfterAddTarget(this)
{
   FALCON_DECLARE_SYN_CLASS( expr_accumulator )
   apply = apply_;
}

   
ExprAccumulator::ExprAccumulator( const ExprAccumulator& other ):
   Expression( other ),
   m_target(0),
   m_filter(0),
   m_vector(0),
   m_stepGenIter(this),
   m_stepGenNext(this),
   m_stepTakeNext(this),
   m_stepAfterFilter(this),
   m_stepAfterAddTarget(this)
{
   if( other.m_target != 0 )
   {
      m_target = other.m_target->clone();
      m_target->setParent(this);
   }

   if( other.m_filter != 0 )
   {
      m_filter = other.m_filter->clone();
      m_filter->setParent(this);
   }

   if( other.m_vector != 0 )
   {
      m_vector = other.m_vector->clone();
      m_vector->setParent(this);
   }

   apply = apply_;
}


ExprAccumulator::~ExprAccumulator()
{
   dispose(m_filter);
   dispose(m_target);
   dispose(m_vector);
}


void ExprAccumulator::render( TextWriter* tw, int32 depth ) const
{
   String temp1, temp2;
   tw->write( renderPrefix(depth) );

   if( depth < 0 )
   {
      tw->write( "(" );
   }

   tw->write("^[" );

   if( m_vector != 0 )
   {
      int len = m_vector->arity();

      for( int i = 0 ; i < len; i++ )
      {
         if( i != 0 )
         {
            tw->write(", ");
         }

         TreeStep* expr = m_vector->nth(i);
         expr->render(tw, relativeDepth(depth) );
      }
   }

   tw->write("]");

   if( m_filter != 0 )
   {
      tw->write(" ");
      m_filter->render(tw, relativeDepth(depth) );
   }

   if( m_target != 0 )
   {
      tw->write(" => ");
      m_target->render(tw, relativeDepth(depth) );
   }

   if( depth >= 0 )
   {
      tw->write("\n");
   }
   else
   {
      tw->write(")");
   }
}


bool ExprAccumulator::simplify( Item& ) const
{
   return false;
}


int32 ExprAccumulator::arity() const
{
   return (int32) 2;
}


TreeStep* ExprAccumulator::selector() const
{
   return m_vector;
}


bool ExprAccumulator::selector( TreeStep* e )
{
   if( e == 0 || (e->category() == TreeStep::e_cat_expression && static_cast<Expression*>(e)->trait() == Expression::e_trait_vectorial && e->setParent(this)) )
   {
      dispose(m_vector);
      m_vector = e;
      return true;
   }

   return false;
}


bool ExprAccumulator::filter( TreeStep* ts )
{
   if( ts == 0 || ts->setParent(this) )
   {
      dispose(m_filter);
      m_filter = ts;
      return true;
   }
   return false;
}


bool ExprAccumulator::target( TreeStep* ts )
{
   if( ts == 0 || ts->setParent(this) )
   {
      dispose(m_target);
      m_target = ts;
      return true;
   }
   return false;
}

TreeStep* ExprAccumulator::nth( int32 n ) const
{
   if( n == -1 || n == 0 ) return filter();
   if( n == -2 || n == 1 ) return target();
   return 0;
}


bool ExprAccumulator::setNth( int32 n, TreeStep* ts )
{
   if( (n == -1 || n == 0) && filter( ts ) ) return true;
   if( (n == -2 || n == 1) && target( ts ) ) return true;
   return false;
}

//=================================================================
// Utilities
//

void ExprAccumulator::addToTarget( VMContext* ctx, Item* base, int32 arity ) const
{
   TRACE("ExprAccumulator::addToTarget %p %p %d", ctx, base, arity );

   ctx->pushCode(&m_stepAfterAddTarget);

   Item toBeAdded;
   if( arity > 1 )
   {
      ItemArray* arr = new ItemArray();
      arr->copyFromData(base, arity);
      toBeAdded = FALCON_GC_HANDLE(arr);
   }
   else {
      toBeAdded = base[0];
   }

   // invoke the target aadd
   Item target = base[arity];
   Class* cls = 0;
   void* data = 0;
   target.forceClassInst(cls, data);
   ctx->pushData(target);
   ctx->pushData(toBeAdded);
   cls->op_aadd(ctx, data);
}


void ExprAccumulator::regress( VMContext* ctx ) const
{
   TRACE("ExprAccumulator::regress %p", ctx);

   //CodeFrame& cf = ctx->currentCode();
   int arity = m_vector->arity();
   Item* base = ctx->opcodeParams( arity * 3 + 2 );
   // it's last element + 1 ...
   Item* top = base + arity;
   while( base < top )
   {
      //... so we decrement it now
      --top;
      if( top->isDoubt() )
      {
         // ask next element
         int32 pos = (int32) ( top-base );
         int toBeDel = (arity - (pos + 1)) * 2;
         ctx->popData( toBeDel );

         // configure the PStepGenIter
         ctx->currentCode().m_seqId = pos;
         ctx->pushCode(&m_stepGenNext);
         return;
      }
   }

   // we're done. Push the target and get away.
   Item target = base[arity];
   ctx->popData( arity*4 + 2 );
   ctx->pushData(target);
   ctx->popCode(); // remove the PStepGenIter
}


//=================================================================
// Apply
//

void ExprAccumulator::apply_( const PStep* ps, VMContext* ctx )
{
   static PStep* psnil = &Engine::instance()->stdSteps()->m_pushNil;
   const ExprAccumulator* self = static_cast<const ExprAccumulator*>(ps);

   // without a vector, we just generate the target or call the filter, if any.
   int32 len;
   if( self->m_vector == 0 || (len = self->m_vector->arity()) == 0 )
   {
      TRACE("ExprAccumulator::without a vector, target=%p, filter=%p", self->m_target, self->m_filter );

      if( self->m_target != 0 )
      {
         ctx->stepIn(self->m_target);
      }
      else if( self->m_filter != 0 )
      {
         ctx->pushData( Item(self->m_filter->handler(), self->m_filter ) );
         self->m_filter->handler()->op_call(ctx, 0, self->m_filter);
      }
      else
      {
         ctx->pushData(Item());
      }

      // we're done
      ctx->popCode();
      return;
   }

   // we generate all the items in the expression vector
   CodeFrame& cf = ctx->currentCode();
   int32& seqId = cf.m_seqId;

   TRACE("ExprAccumulator::apply_ at step %d/%d -- dataDepth: %d", seqId, len, (int)ctx->dataSize() );

   while( seqId < len )
   {
      TreeStep* gen = self->m_vector->nth(seqId++);
      if( ctx->stepInYield(gen) )
      {
         return;
      }
   }

   // we're done
   ctx->popCode();
   /*
   if( self->m_filter == 0 && self->m_target == 0 )
   {
      // without filter or target, just generate this stuff as an array
      ItemArray* arr = new ItemArray;
      arr->copyFromData( ctx->opcodeParams(len), len );
      ctx->popData(len);
      ctx->pushData(FALCON_GC_HANDLE(arr));
      return;
   }
   */

   // now push the space needed for the buffer
   ctx->addSpace(len);

   // next step will be that of generating iterators.
   ctx->pushCode( &self->m_stepGenIter );

   // Then, generate the filter
   if( self->m_filter != 0 )
   {
      ctx->pushCode( self->m_filter );
   }
   else
   {
      ctx->pushCode( psnil );
   }

   // first of all, generate the target
   if( self->m_target != 0 )
   {
      ctx->pushCode( self->m_target );
   }
   else
   {
      ctx->pushCode( psnil );
   }
}


void ExprAccumulator::PStepGenIter::apply_( const PStep* ps, VMContext* ctx )
{
   // generate all the iterators.
   const ExprAccumulator::PStepGenIter* giself = static_cast<const ExprAccumulator::PStepGenIter*>(ps);
   const ExprAccumulator* self = giself->m_owner;

   // Don't pop: we need to stay there to account for where we're at now.
   CodeFrame& cf = ctx->currentCode();
   int32 arity = self->m_vector->arity();
   int32 pos = cf.m_seqId; // this is the element we're about

   TRACE("ExprAccumulator::PStepGenIter::apply_ at step %d/%d -- dataDepth: %d", pos, arity, (int)ctx->dataSize() );

   ctx->pushCode( &self->m_stepGenNext );

   // get an iterator for the generator
   Item* base = ctx->opcodeParams( (arity*2) +2 + (pos*2) );
   Item sequence = base[pos];
   Class* cls = 0;
   void* data = 0;
   sequence.forceClassInst(cls, data);
   ctx->pushData( sequence ); // push the sequence
   cls->op_iter(ctx, data); // generate the iterator
}


void ExprAccumulator::PStepGenNext::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprAccumulator::PStepGenNext* nself = static_cast<const ExprAccumulator::PStepGenNext*>(ps);
   const ExprAccumulator* self = nself->m_owner;

   // we won't be back anymore (unless re-pushed)
   ctx->popCode();
   CodeFrame& cf = ctx->currentCode();

   // is the iterator ok?
   int32 seqId = cf.m_seqId;
   int32 arity = self->m_vector->arity();

   TRACE("ExprAccumulator::PStepGenNext::apply_ at step %d/%d -- dataDepth: %d", seqId, arity, (int)ctx->dataSize() );
   if( ctx->topData().isNil() ) // nil is the non-iterator.
   {
      // roll back the iter operation:
      ctx->popCode();
      ctx->popData( arity*2 + seqId * 2 + 1 );
      // 1 extra element is left on the stack:
      ctx->topData().setNil();
      return;
   }

   // get our next element (same id as ours)
   ctx->pushCode( &self->m_stepTakeNext );

   Item& sequence = ctx->opcodeParam( 1 );
   Class* cls = 0;
   void* data = 0;
   sequence.forceClassInst(cls, data);
   cls->op_next(ctx, data); // generate the iterator
}


void ExprAccumulator::PStepTakeNext::apply_( const PStep* ps, VMContext* ctx )
{
   const ExprAccumulator::PStepTakeNext* nself = static_cast<const ExprAccumulator::PStepTakeNext*>(ps);
   const ExprAccumulator* self = nself->m_owner;

   // we're not needed again.
   ctx->popCode();

   // Get the id of the base PStepGenIter
   CodeFrame& cf = ctx->currentCode();
   int32 arity = self->m_vector->arity();
   int32 seqId = cf.m_seqId; // note: we get it by value
   TRACE("ExprAccumulator::PStepTakeNext::apply_ at step %d/%d -- dataDepth: %d", seqId, arity, (int)ctx->dataSize() );

   // is the item ok?
   if( ctx->topData().isBreak() )
   {
      // roll back the operation (there's a PStepGenIter in)
      ctx->popCode();
      ctx->popData( (seqId+1)*2 + 1 + 1 ); // remove the generated stack iterators and filter + generated.
      Item target = ctx->topData();    // get the target
      ctx->popData(arity*2);             // remove the buffer+generated minus last
      // 1 extra element is left on the stack:
      ctx->topData() = target;
      return;
   }

   // move the item in place.
   // +1 = target; +1 = generated item.
   Item top = ctx->topData();
   ctx->popData();
   Item* base = ctx->opcodeParams( arity + 2 + ((seqId+1) * 2) );
   base[seqId] = top;

   // prepare to push next iterations
   ++seqId;

   // are we done with getting the items?
   if( seqId == arity )
   {
      if ( self->m_filter != 0 )
      {
         ctx->pushCode(&self->m_stepAfterFilter);
         Item filter = base[arity+1];
         // prepare the call space
         ctx->pushData(filter);
         ctx->addSpace(arity);

         // 1 arity for the buffer copy
         // 2 for target & filter
         // arity* 2 for the iterator space
         // 1 for the call item
         // arity for the final copy.
         ctx->copyData(ctx->opcodeParams(arity), arity, ctx->dataSize() - (arity*4 + 3) );

         // do the call
         Class* cls = 0;
         void* data = 0;
         filter.forceClassInst(cls, data);
         cls->op_call(ctx, arity, data);
      }
      else if( self->m_target != 0 )
      {
         self->addToTarget( ctx, base, arity );
      }
      else
      {
         /*
         // just store the new array in a master array
         Item& target = base[arity];
         ItemArray* masterArr;
         if( ! target.isArray() )
         {
            masterArr = new ItemArray();
            target = FALCON_GC_HANDLE(masterArr);
         }
         else {
            masterArr = target.asArray();
         }

         if( arity > 1 )
         {
            ItemArray* arr = new ItemArray();
            arr->copyFromData(base, arity);
            masterArr->append(FALCON_GC_HANDLE(arr));
         }
         else
         {
            masterArr->append( base[0] );
         }
         */

         // try another iteration
         self->regress(ctx);
      }
   }
   else
   {
      // let the PStepGetIter to get the next iter.
      ctx->currentCode().m_seqId = seqId;
   }
}


void ExprAccumulator::PStepAfterFilter::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE("ExprAccumulator::PStepAfterFilter -- dataDepth: %d", (int) ctx->dataSize() );
   const ExprAccumulator::PStepAfterFilter* nself = static_cast<const ExprAccumulator::PStepAfterFilter*>(ps);
   const ExprAccumulator* self = nself->m_owner;

   // not needed anymore
   ctx->popCode();

   int32 arity = self->m_vector->arity();
   Item top = ctx->topData();
   ctx->popData();

   if( top.isBreak() )
   {
      // roll back the operation (there's a PStepGenIter in)
      ctx->popCode();
      ctx->popData( arity * 2 + 1 );
      Item target = ctx->topData();
      ctx->popData(arity*2);
      // 1 extra element is left on the stack:
      ctx->topData() = target;
      return;
   }
   else if( self->m_target != 0 )
   {
      if( top.isOob() )
      {
         top.setOob(false);
         ctx->pushCode(&self->m_stepAfterAddTarget);

         // invoke the target aadd
         Item target = *ctx->opcodeParams( arity* 2 + 2 );
         Class* cls = 0;
         void* data = 0;
         target.forceClassInst(cls, data);
         ctx->pushData(target);
         ctx->pushData(top);
         cls->op_aadd(ctx, data);
         return;
      }
      else if( top.isTrue() )
      {
         Item* base = ctx->opcodeParams( arity* 3 + 2 );
         self->addToTarget( ctx, base, arity );
         return;
      }
   }
   // In any other case, just regress
   self->regress(ctx);
}


void ExprAccumulator::PStepAfterAddTarget::apply_( const PStep* ps, VMContext* ctx )
{
   TRACE("ExprAccumulator::PStepAfterAddTarget -- dataDepth: %d", (int) ctx->dataSize() );

   const ExprAccumulator::PStepAfterAddTarget* nself = static_cast<const ExprAccumulator::PStepAfterAddTarget*>(ps);
   const ExprAccumulator* self = nself->m_owner;

   ctx->popCode();
   // flat data?
   if( ! ctx->topData().isUser() )
   {
      // we need to save it
      int32 arity = self->m_vector->arity();
      ctx->opcodeParam( arity * 2 + 2 ) = ctx->topData();
   }
   ctx->popData(); // pop the self left by the target += x
   self->regress(ctx);
}

}

/* end of expraccumulator.cpp */

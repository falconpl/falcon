/*
   FALCON - The Falcon Programming Language.
   FILE: sequence.cpp

   Definition of abstract sequence class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 13 Jul 2009 23:00:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/sequence.h>
#include <falcon/vm.h>
#include <falcon/error.h>
#include <falcon/garbageable.h>
#include <falcon/garbagepointer.h>
#include <falcon/rangeseq.h>

namespace Falcon {

class FALCON_DYN_CLASS GarbagePointer2: public GarbagePointer
{
   FalconData *m_ptr;

public:
   GarbagePointer2( FalconData *p ):
         GarbagePointer( p )
   {
   }

   virtual ~GarbagePointer2() {}
   virtual bool finalize() { return false; }
};

//===================================================================
// TODO Remvoe at first major release
//

inline bool s_appendMe( VMachine *vm, Sequence* me, const Item &source, const Item &filter )
{
   if( filter.isNil() )
   {
      me->append( source );
   }
   else
   {
      vm->pushParam( source );
      vm->pushParam( vm->self() );
      vm->callItemAtomic(filter,2);
      if ( ! vm->regA().isOob() )
         me->append( vm->regA().isString() ? new CoreString( *vm->regA().asString() ) : vm->regA() );
      else if ( vm->regA().isInteger() && vm->regA().asInteger() == 0 )
         return false;
   }

   return true;
}

void Sequence::comprehension( VMachine* vm, const Item& cmp, const Item& filter )
{
   if( cmp.isRange() )
   {
      {
         if ( cmp.asRangeIsOpen() )
         {
            throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "open range" ) );
         }

         int64 start = cmp.asRangeStart();
         int64 end = cmp.asRangeEnd();
         int64 step = cmp.asRangeStep();
         if ( start == end ) {
            if ( step < 0 )
            {
               s_appendMe( vm, this, start, filter );
            }
            return;
         }

         if( start < end )
         {
            if ( step < 0 )
               return;
            if ( step == 0 )
               step = 1;

            while( start < end )
            {
               if ( ! s_appendMe( vm, this, start, filter ) )
                  break;
               start += step;
            }
         }
         else {
            if ( step > 0 )
               return;
            if ( step == 0 )
               step = -1;

            while( start >= end )
            {
               if ( ! s_appendMe( vm, this, start, filter ) )
                  break;
               start += step;
            }
         }
      }
   }
   else if ( cmp.isCallable() )
   {
      while( true )
      {
         vm->callItemAtomic( cmp, 0 );
         if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
         {
            return;
         }

         Item temp = vm->regA();
         if( ! s_appendMe( vm, this, temp, filter ) )
            break;
      }
   }
   // todo --- remove this as soon as we have iterators on ItemArrays
   else if ( cmp.isArray() )
   {
      const CoreArray& arr = *cmp.asArray();

      for( uint32 i = 0; i < arr.length(); i ++ )
      {
         if ( ! s_appendMe( vm, this, arr[i], filter ) )
            break;
      }
   }
   else if ( (cmp.isObject() && cmp.asObjectSafe()->getSequence() ) )
   {
      //Sequence* seq = cmp.isArray() ? &cmp.asArray()->items() : cmp.asObjectSafe()->getSequence();

      Sequence* seq = cmp.asObjectSafe()->getSequence();
      Iterator iter( seq );
      while( iter.hasCurrent() )
      {
         if ( ! s_appendMe( vm, this, iter.getCurrent(), filter ) )
            break;
         iter.next();
      }
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "A|C|R|Sequence, [C]" ) );
   }
}

//==========================================================================
//

static bool multi_comprehension_generic_single_loop( VMachine* vm );
static bool multi_comprehension_callable_multiple_loop( VMachine* vm );
static bool multi_comprehension_filtered_loop( VMachine* vm );

static bool comp_get_all_items_callable_next( VMachine *vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - Sequence GC pointer
   // 2 - filter (nil)
   // 3 - Source (callable)

   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're done.
      vm->retval(vm->self());
      return false;
   }

   // add the data.
   dyncast<Sequence*>(vm->local(1)->asGCPointer())->append(
         vm->regA().isString() ? new CoreString( *vm->regA().asString() ) : vm->regA() );

   // iterate.
   vm->callFrame( *vm->local(3), 0 );
   return true;
}

// gets all the items that it can from a comprehension
static bool comp_get_all_items( VMachine *vm, const Item& cmp )
{
   Sequence* sequence = dyncast<Sequence*>(vm->local(1)->asGCPointer());

   if ( cmp.isRange() )
   {
      if ( cmp.asRangeIsOpen() )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin( e_orig_runtime )
            .extra( "open range" ) );
      }

      int64 start = cmp.asRangeStart();
      int64 end = cmp.asRangeEnd();
      int64 step = cmp.asRangeStep();

      if ( start == end )
      {
         if ( step < 0 )
         {
            sequence->append( start );
         }
         return false; // all done.
      }

      if( start < end )
      {
         if ( step < 0 )
            return true;
         if ( step == 0 )
            step = 1;

         while( start < end )
         {
            sequence->append( start );
            start += step;
         }
      }
      else {
         if ( step > 0 )
            return false;
         if ( step == 0 )
            step = -1;

         while( start >= end )
         {
            sequence->append( start );
            start += step;
         }
      }
   }
   else if ( cmp.isCallable() )
   {
      // change the frame handler, so that instead of having
      vm->returnHandler( comp_get_all_items_callable_next );

      vm->callFrame( cmp, 0 );
      // need more calls.
      return true;
   }
   else if ( cmp.isArray() )
   {
      const CoreArray& arr = *cmp.asArray();
      for( uint32 i = 0; i < arr.length(); i ++ )
      {
         sequence->append( arr[i].isString() ? new CoreString( *arr[i].asString() ) : arr[i] );
      }
   }
   else if ( (cmp.isObject() && cmp.asObjectSafe()->getSequence() ) )
   {
      Sequence* origseq = cmp.asObjectSafe()->getSequence();
      Iterator iter( origseq );
      while( iter.hasCurrent() )
      {
         sequence->append( iter.getCurrent().isString() ? new CoreString( *iter.getCurrent().asString() ) : iter.getCurrent() );
         iter.next();
      }
   }
   else {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "A|C|R|Sequence, [C]" ) );
   }

   // we have processed all the sequence.
   vm->retval( vm->self() );
   return false;
}

#if 0

static bool multi_comprehension_generic_multiple_loop_post_filter( VMachine* vm )
{
   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're done.
      vm->retval( vm->self() );
      return false;
   }

   // TODO: Change in a generic sequence
   dyncast<Sequence*>(vm->local(1)->asGCPointer())->append( vm->regA() );
   vm->returnHandler( &multi_comprehension_generic_multiple_loop );
   return true;
}


static bool multi_comprehension_generic_multiple_loop_post_call( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GC Ptr to the sequence
   // 2 - filter
   // 3 - first iterator or callable
   // 4 - second iterator or callable
   // N-3 - ... nth iterator or callable.
   // <Sequence where to store the data>

   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're done.
      vm->retval( vm->self() );
      return false;
   }

   if( vm->local(1)->isNil() )
   {
      // no filter
      vm->local( vm->currentFrame()->stackSize()-1 )->asArray()->append( vm->regA() );
      vm->returnHandler( &multi_comprehension_generic_multiple_loop_post_filter );
   }
   else
   {
      vm->returnHandler( &multi_comprehension_generic_multiple_loop_post_filter );
      Item filter = *vm->local(2);
      vm->pushParam( vm->regA() );
      vm->pushParam( vm->self() );
      vm->callFrame( filter, 2 );
   }

   return true;
}


static bool multi_comprehension_generic_multiple_loop( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GCPointer to the sequence
   // 2 - filter
   // 3 - first iterator
   // 4 - second iterator
   // N-3 - ... nth iterator
   // <Sequence where to store the data>

   uint32 ssize = vm->currentFrame()->stackSize()-4;
   int64 current = vm->local(0)->asInteger();

   // prepare next loop
   int64 next = current + 1;
   if( next > (int64) ssize )
   {
      Item item = *vm->local(ssize + 3);
      *vm->local(ssize + 3) = new CoreArray( ssize );

      // do we have a filter?
      if( ! vm->local(2)->isNil() )
      {
         // prepare for the next loop
         *vm->local(0) = (int64) 0;

         // call the filter
         Item filter = *vm->local(2);
         vm->returnHandler( &multi_comprehension_generic_multiple_loop_post_filter );
         vm->pushParam( item );
         vm->pushParam( vm->self() );
         vm->callFrame( filter, 2 );

         return true;
      }
      else
      {
         dyncast<Sequence*>(vm->local(1)->asGCPointer())->append( item );
      }

      next = 1;
      current = 0;
   }
   *vm->local(0) = next;

   // get the element
   Item* src = vm->local(current+3);

   if ( src->isOob() )
   {
      // it's a callable -- we must call it.

      // change the next loop
      vm->returnHandler( &multi_comprehension_generic_multiple_loop_post_call );
      vm->callFrame( *src, 0 );

   }
   else
   {
      Iterator* iter = dyncast<Iterator*>( src->asGCPointer() );
      if ( iter->hasCurrent() )
      {
         vm->local(ssize + 3)->asArray()->append( iter->getCurrent() );
         iter->next();
      }
      else
      {
         // we're done; we must discard this work-in-progress
         vm->retval( vm->self() );
         return false;
      }
   }

   // continue
   return true;
}


static bool multi_comprehension_generic_multiple_loop( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GCPointer to the sequence
   // 2 - filter
   // 3 - first iterator or callable
   // 4 - second iterator or callable
   // N-3 - ... nth iterator pr callable
   // <Sequence where to store the data>

   uint32 ssize = vm->currentFrame()->stackSize()-4;
   int64 current = vm->local(0)->asInteger();

   // prepare next loop
   int64 next = current + 1;
   if( next > (int64) ssize )
   {
      Item item = *vm->local(ssize + 3);
      *vm->local(ssize + 3) = new CoreArray( ssize );

      // do we have a filter?
      if( ! vm->local(2)->isNil() )
      {
         // prepare for the next loop
         *vm->local(0) = (int64) 0;

         // call the filter
         Item filter = *vm->local(2);
         vm->returnHandler( &multi_comprehension_generic_multiple_loop_post_filter );
         vm->pushParam( item );
         vm->pushParam( vm->self() );
         vm->callFrame( filter, 2 );

         return true;
      }
      else
      {
         dyncast<Sequence*>(vm->local(1)->asGCPointer())->append( item );
      }

      next = 1;
      current = 0;
   }
   *vm->local(0) = next;

   // get the element
   Item* src = vm->local(current+3);

   Iterator* iter = dyncast<Iterator*>( src->asGCPointer() );
   if ( iter->hasCurrent() )
   {
      vm->local(ssize + 3)->asArray()->append( iter->getCurrent() );
      iter->next();
   }
   else
   {
      // we're done; we must discard this work-in-progress
      vm->retval( vm->self() );
      return false;
   }

   // continue
   return true;
}

#endif

static bool multi_comprehension_filtered_loop_next( VMachine* vm )
{
   Sequence* self = dyncast<Sequence*>(vm->local(1)->asGCPointer());

   Item& regA = vm->regA();
   // do the last operation was an oob?
   if( regA.isOob() )
   {
      // is it an integer?
      if( regA.isInteger() )
      {
         // Request to stop?
         if ( regA.asInteger() == 0 )
         {
            vm->retval( vm->self() );
            return false;
         }
         else if ( regA.asInteger() != 1 )
         {
            self->append( vm->regA() );
         }
      }
      else
         self->append( vm->regA() );
   }
   else
      self->append( vm->regA().isString() ? new CoreString( *vm->regA().asString() ) : vm->regA() );

   return multi_comprehension_filtered_loop( vm );
}


static bool multi_comprehension_filtered_loop( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GCPointer to the sequence
   // 2 - filter
   // 3 - first iterator
   // 4 - second iterator
   // N-3 - ... nth iterator

   uint32 ssize = vm->currentFrame()->stackSize()-4;

   // create the object to be added.
   for( uint32 elem = 0; elem < ssize; ++elem )
   {
      Iterator* seq = dyncast<Iterator*>(vm->local(elem+3)->asGCPointer());
      if( ! seq->hasCurrent() )
      {
         vm->retval( vm->self() );
         return false;
      }

      vm->pushParam( seq->getCurrent() );
   }

   // advance
   uint32 pos = ssize;
   while( pos > 0 )
   {
      Iterator* seq = dyncast<Iterator*>(vm->local(pos-1+3)->asGCPointer());

      // can we advance?
      if( seq->next() )
         break;

      //--- and advance the previous element.
      --pos;

      //--- no? reset this element,
      if( pos > 0 )
      {
         // but only if it's not the first. Then, we leave this set.
         seq->goTop();
         // leaving the first element set at bottom, we'll terminate at next loop
      }
   }

   vm->pushParam( vm->self() );
   vm->returnHandler( multi_comprehension_filtered_loop_next );
   vm->callFrame( *vm->local(2), ssize + 1);

   // continue
   return true;
}


static bool multi_comprehension_generic_single_loop_post_filter( VMachine* vm )
{
   if( vm->regA().isOob() && vm->regA().isInteger() )
   {
      if( vm->regA().asInteger() == 0 )
      {
         // we're done.
         vm->retval( vm->self() );
         return false;
      }
      else if ( vm->regA().asInteger() == 1 )
      {
         // don't append
         vm->returnHandler( multi_comprehension_generic_single_loop );
         return true;
      }
   }

   dyncast<Sequence*>(vm->local(1)->asGCPointer())->append(
         vm->regA().isString() ?
               new CoreString( *vm->regA().asString() ) : vm->regA() );
   vm->returnHandler( multi_comprehension_generic_single_loop );
   return true;
}

static bool multi_comprehension_generic_single_loop_post_call( VMachine* vm )
{
   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're done.
      vm->retval( vm->self() );
      return false;
   }

   // we know we have a filter, or we'd be in the simpler get all case.
   Item filter = *vm->local(2);
   vm->returnHandler( multi_comprehension_generic_single_loop_post_filter );
   vm->pushParam( vm->regA() );
   vm->pushParam( vm->self() );
   vm->callFrame( filter, 2 );

   return true;
}

static bool multi_comprehension_generic_single_loop( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GCPointer of the sequence
   // 2 - filter
   // 3 - iterator or callable

   // get the element
   Item* src = vm->local(3);

   if ( src->isOob() )
   {
      // it's a callable -- we must call it.
      // change the next loop
      vm->returnHandler( &multi_comprehension_generic_single_loop_post_call );
      vm->callFrame( *src, 0 );
   }
   else
   {
      Iterator* iter = dyncast<Iterator*>( src->asGCPointer() );
      if ( iter->hasCurrent() )
      {
         Item item = iter->getCurrent();
         iter->next();

         // call the filter --  we know we have it or we'd be in the simpler get all case
         Item filter = *vm->local(2);
         vm->returnHandler( &multi_comprehension_generic_single_loop_post_filter );
         vm->pushParam( item );
         vm->pushParam( vm->self() );
         vm->callFrame( filter, 2 );
      }
      else
      {
         // we're done; we must discard this work-in-progress
         vm->retval( vm->self() );
         return false;
      }
   }

   // continue
   return true;
}


static bool multi_comprehension_callable_multiple_loop_next( VMachine* vm )
{
   uint32 ssize = vm->currentFrame()->stackSize();
   Item* array = vm->local( ssize-1 );
   CoreArray* ca = array->asArray();
   int64 current = vm->local(0)->asInteger();

   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      // we're done with this callable. We must store the result in the right place
      vm->local( current + 3 )->setGCPointer( new GarbagePointer2( new Iterator( &ca->items()) ) );
      vm->local(0)->setInteger( current+1 );
      vm->returnHandler( multi_comprehension_callable_multiple_loop );
   }
   else
   {
      ca->append( vm->regA().isString() ? new CoreString( *vm->regA().asString() ) : vm->regA() );
      vm->callFrame( *vm->local(current+3), 0 );
   }

   return true;
}


static void multi_comprehension_generate_all( VMachine* vm )
{
   uint32 ssize = vm->currentFrame()->stackSize()-4;
   Sequence* self = dyncast<Sequence*>(vm->local(1)->asGCPointer());

   while( true )
   {
      // create the object to be added.
      CoreArray* cret = new CoreArray( ssize );
      for( uint32 elem = 0; elem < ssize; ++elem )
      {
         Iterator* seq = dyncast<Iterator*>(vm->local(elem+3)->asGCPointer());
         if( ! seq->hasCurrent() )
            return;
         cret->append( seq->getCurrent().isString() ? new CoreString( *seq->getCurrent().asString() ) : seq->getCurrent() );
      }

      // append it
      self->append( cret );

      // advance
      uint32 pos = ssize;
      while( pos > 0 )
      {
         Iterator* seq = dyncast<Iterator*>(vm->local(pos-1+3)->asGCPointer());

         // can we advance?
         if( seq->next() )
            break;

         //--- no? reset this element,
         seq->goTop();
         //--- and advance the previous element.
         --pos;
      }

      // did we reset the topmost element?
      if ( pos == 0)
         break;
   }

   vm->retval( vm->self() );
}


static bool multi_comprehension_callable_multiple_loop( VMachine* vm )
{
   uint32 ssize = vm->currentFrame()->stackSize()-4;
   int64 current = vm->local(0)->asInteger();

   while( (! vm->local(current+3)->isOob()) && current < ssize )
   {
      ++current;
   }

   if( current == ssize )
   {
      // No filter ?
      if( vm->local(2)->isNil() )
      {
         multi_comprehension_generate_all( vm );
         return false;
      }

      // we have run all the runnable generators. Now it's time to pack this up
      vm->returnHandler( multi_comprehension_filtered_loop );

      // it's useless to wait -- also call it now
      return multi_comprehension_filtered_loop( vm );
   }

   Item callable = *vm->local(current+3);
   // prepare for the next loop
   vm->local(0)->setInteger( current );

   // ready to accept the reutrn of the function
   vm->local( ssize + 3 )->setArray( new CoreArray );
   vm->returnHandler( multi_comprehension_callable_multiple_loop_next );

   // call it
   vm->callFrame( callable, 0 );

   return true;
}


static bool multi_comprehension_first_loop( VMachine* vm )
{
   // STACK LOCAL :
   // 0 - Global counter
   // 1 - GCPointer of the sequence
   // 2 - filter
   // 3 - first comp source
   // 4 - second cmp source
   // N-3 - ... nth source.

   uint32 ssize = vm->currentFrame()->stackSize();
   if ( ssize < 4 )
      return false;
   
   uint32 sources = ssize - 3;
   if( sources == 1 && vm->local(2)->isNil() )
   {
      // we can use the simplified single comprehension system.
      return comp_get_all_items( vm, *vm->local(3) );
   }

   bool hasCallable = false;

   // No luck; let's proceed with next loop
   // we must transform all the sources in their iterator,
   // For callable elements, we just set OOB.
   for ( uint32 nSrc = 0; nSrc < sources; nSrc ++ )
   {
      Item* src = vm->local( 3 + nSrc );
      if( src->isCallable() )
      {
         src->setOob(true);
         hasCallable = true;
      }
      else
      {
         src->setOob(false); // just in case

         if ( src->isRange() )
         {
            // the iterator will keep alive the range sequence as long as it exists.
            src->setGCPointer( new GarbagePointer2(
                  new Iterator( new RangeSeq( *src->asRange() ) ) ) );
         }
         else if ( src->isArray() )
         {
            src->setGCPointer( new GarbagePointer2(
                     new Iterator( &src->asArray()->items() ) ) );
         }
         else if ( (src->isObject() && src->asObjectSafe()->getSequence() ) )
         {
            src->setGCPointer( new GarbagePointer2(
                  new Iterator( src->asObjectSafe()->getSequence() ) ) ) ;
         }
         else {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                     .origin( e_orig_runtime )
                     .extra( "A|C|R|Sequence, [C]" ) );
         }
      }
   }

   if( sources == 1 )
   {
      vm->returnHandler( &multi_comprehension_generic_single_loop );

      // it's useless to wait -- also call it now
      return multi_comprehension_generic_single_loop( vm );
   }
   else
   {
      // Todo: add an empty copy of the sequence in self instead.
      vm->pushParam( Item() );

      if( ! hasCallable )
      {
         // No filter ?
         if( vm->local(2)->isNil() )
         {
            multi_comprehension_generate_all( vm );
            return false;
         }

         vm->returnHandler( &multi_comprehension_filtered_loop );

         // it's useless to wait -- also call it now
         return multi_comprehension_filtered_loop( vm );
      }
      else
      {
         vm->returnHandler( multi_comprehension_callable_multiple_loop );

         // it's useless to wait -- also call it now
         return multi_comprehension_callable_multiple_loop( vm );
      }
   }
}


bool Sequence::comprehension_start( VMachine* vm, const Item& self, const Item& filter )
{
   if( ! (filter.isNil() || filter.isCallable()) )
   {
      throw new ParamError( ErrorParam( e_param_type, __LINE__ )
               .origin( e_orig_runtime )
               .extra( "filter" ) );
   }

   // local copy before changing the stack.
   Item copyFilter = filter;
   Item selfCpy = self;

   // Ask for a new stack frame, with immediate invocation of the return frame.
   vm->invokeReturnFrame(multi_comprehension_first_loop);
   // prepare data stub
   vm->addLocals( 3 );
   vm->self() = selfCpy;
   *vm->local(0) = (int64) 0;    // global counter
   vm->local(1)->setGCPointer( new GarbagePointer2( this ) );
   *vm->local(2) = filter;       // filter (may be nil)

   return true;
}

void Sequence::gcMark( uint32 gen )
{
   if ( m_owner != 0 && m_owner->mark() != gen )
      m_owner->gcMark( gen );
}


void Sequence::invalidateAllIters()
{
   while( m_iterList != 0 )
   {
      m_iterList->invalidate();
      m_iterList = m_iterList->nextIter();
   }
}


void Sequence::invalidateAnyOtherIter( Iterator* iter )
{
   // is the iterator really in our list?
   bool foundMe = false;
   
   while( m_iterList != 0 )
   {
      if ( m_iterList != iter )
      {
         m_iterList->invalidate();
      }
      else
         foundMe = true;
         
      m_iterList = m_iterList->nextIter();      
   }
   
   //... then save it and set it at the only iterator.
   fassert( foundMe );  // actually, it should be...
   
   if ( foundMe ) 
   {
      iter->nextIter( 0 );
      m_iterList = iter;
   }
}


void Sequence::getIterator( Iterator& tgt, bool tail ) const
{
   tgt.sequence( const_cast<Sequence*>(this) );
   tgt.nextIter( m_iterList );
   m_iterList = &tgt;   
}


void Sequence::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   tgt.sequence( const_cast<Sequence*>(this) );
   tgt.nextIter( m_iterList );
   m_iterList = &tgt;   
}


void Sequence::disposeIterator( Iterator& tgt ) const
{
   Iterator *iter = m_iterList;
   Iterator *piter = 0;
   
   while( iter != 0 )
   {
      if ( iter == &tgt )
      {
         // found!
         if ( piter == 0) {
            // was the first one!
            m_iterList = iter->nextIter();
         }
         else {
            piter->nextIter( iter->nextIter() );
         }
         
         iter->invalidate();
         return;
      }
      
      piter = iter;
      iter = iter->nextIter();
   }  
   
   // we should have found an iterator of ours
   fassert( false );
}

void Sequence::invalidateIteratorOnCriterion() const
{
   Iterator *iter = m_iterList;
   Iterator *piter = 0;
   
   while( iter != 0 )
   {
      if ( onCriterion( iter ) )
      {
         // found!
         if ( piter == 0) {
            // was the first one!
            m_iterList = iter->nextIter();
         }
         else {
            piter->nextIter( iter->nextIter() );
         }
         
         iter->invalidate();
         Iterator* old = iter;
         iter = iter->nextIter();
         old->nextIter( 0 );
         continue;
      }
      
      piter = iter;
      iter = iter->nextIter();
   }  
}

}

/* end of sequence.cpp */

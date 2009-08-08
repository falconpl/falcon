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

namespace Falcon {

inline bool s_appendMe( VMachine *vm, Sequence* me, const Item &source, const Item &filter )
{
   if( filter.isNil() )
   {
      me->append( source );
   }
   else
   {
      vm->pushParameter( source );
      vm->pushParameter( vm->self() );
      vm->callItemAtomic(filter,2);
      if ( ! vm->regA().isOob() )
         me->append( vm->regA() );
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

/*
   FALCON - The Falcon Programming Language
   FILE: detmempool.cpp

   Deterministic memory pool
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom gen 28 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Deterministic memory pool definition.
*/

#include <falcon/setup.h>
#include <falcon/detmempool.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>


namespace Falcon {


bool DetMemPool::gcDetMark()
{
   // first, invert mark bit.
   changeMark();

   // presume that all the registers need fresh marking
   markItemFast( m_owner->regA() );
   markItemFast( m_owner->regB() );
   markItemFast( m_owner->self() );
   markItemFast( m_owner->sender() );

   if( m_msTarget < Sys::_milliseconds() )
   {
      // put old mark back in place
      changeMark();
      return false;
   }

   // mark the global symbols
   // When generational gc will be on, this won't be always needed.
   MapIterator iter = m_owner->liveModules().begin();
   while( iter.hasCurrent() )
   {
      LiveModule *currentMod = *(LiveModule **) iter.currentValue();
      currentMod->mark( currentMark() );
      m_aliveMem += sizeof( LiveModule );
      m_aliveItems++;

      ItemVector *current = &currentMod->globals();
      for( uint32 j = 0; j < current->size(); j++ )
         markItemFast( current->itemAt( j ) );

      current = &currentMod->wkitems();
      for( uint32 k = 0; k < current->size(); k++ )
         markItemFast( current->itemAt( k ) );

      if( m_msTarget < Sys::_milliseconds() )
      {
         // put old mark back in place
         changeMark();
         return false;
      }

      iter.next();
   }

   // mark all the items in the coroutines.
   ListElement *ctx_iter = m_owner->getCtxList()->begin();
   uint32 pos;
   ItemVector *stack;
   while( ctx_iter != 0 )
   {
      VMContext *ctx = (VMContext *) ctx_iter->data();

      markItemFast( ctx->regA() );
      markItemFast( ctx->regB() );
      markItemFast( ctx->self() );
      markItemFast( ctx->sender() );

      if( m_msTarget < Sys::_milliseconds() )
      {
         // put old mark back in place
         changeMark();
         return false;
      }

      stack = ctx->getStack();
      for( pos = 0; pos < stack->size(); pos++ ) {
         markItemFast( stack->itemAt( pos ) );
      }

      if( m_msTarget < Sys::_milliseconds() )
      {
         // put old mark back in place
         changeMark();
         return false;
      }

      ctx_iter = ctx_iter->next();
   }

   return true;
}


void DetMemPool::gcDetSweep()
{
   Garbageable *ring = ringRoot();
   if( ring == 0 )
      return;
   Garbageable *ring2 = ring->nextGarbage();
   while( ring2 != ring ) {
      if ( ring2->mark() != currentMark() ) {
         ring2 = ring2->nextGarbage();
         destroyGarbage( ring2->prevGarbage() );
         if( m_msTarget < Sys::_milliseconds() )
         {
            // let's surrender here.
            return;
         }
      }
      else
         ring2 = ring2->nextGarbage();
      // don't check at every loop.
   }
   if ( ring->mark() != currentMark() )
      destroyGarbage( ring );
}


bool DetMemPool::performGC()
{
   // to avoid useless ifs around, in case timeout is zero call the base GC
   if ( m_msLimit == 0 )
      return MemPool::performGC();

   // we are due to complete in...
   m_msTarget = Sys::_milliseconds() + m_msLimit;

   m_aliveItems = 0;
   m_aliveMem = 0;

   // cannot perform?
   if ( ! gcDetMark() )
      return false;

   // is the memory enought to be reclaimed ?
   if ( (m_aliveMem - m_allocatedMem) > m_thresholdReclaim )
   {
      gcDetSweep();
      m_thresholdMemory = m_aliveMem * 2;
   }
   else {
      // it's useful to increase the threshold memory so that we
      // won't be called again too sun.
      m_thresholdMemory *= 2;
   }

   if ( m_thresholdMemory < m_setThreshold )
      m_thresholdMemory = m_setThreshold;

   return true;
}
}


/* end of detmempool.cpp */

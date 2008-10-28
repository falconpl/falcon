/*
   FALCON - The Falcon Programming Language.
   FILE: mempool.cpp

   Memory management system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-03

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/item.h>
#include <falcon/string.h>
#include <falcon/cobject.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cclass.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/membuf.h>
#include <falcon/garbagepointer.h>

// By default, 1MB
#define TEMP_MEM_THRESHOLD 1000000
namespace Falcon {

MemPool::MemPool():
   m_setThreshold( 0 ),
   m_msLimit( 0 ),
   m_status( 0 ),
   m_aliveItems( 0 ),
   m_aliveMem( 0 ),
   m_allocatedItems( 0 ),
   m_allocatedMem( 0 ),
   m_autoClean( true )
{
   m_garbageRoot = 0;
   m_gstrRoot = 0;
   m_availPoolRoot = 0;

   m_thresholdMemory = TEMP_MEM_THRESHOLD;
   m_setThreshold = TEMP_MEM_THRESHOLD;
   m_thresholdReclaim = TEMP_MEM_THRESHOLD/3;
}

MemPool::~MemPool()
{
   if ( m_availPoolRoot != 0 )
   {
      GarbageLock *g1 = m_availPoolRoot;
      GarbageLock *g2 = m_availPoolRoot->next();
      while( g2 != m_availPoolRoot ) {
         g1 = g2;
         g2 = g2->next();
         delete  g1 ;
      }
      delete m_availPoolRoot;
   }

   if ( m_garbageRoot != 0 ) {
      Garbageable *g1 = m_garbageRoot;
      Garbageable *g2 = m_garbageRoot->nextGarbage();
      while( g2 != m_garbageRoot ) {
         g1 = g2;
         g2 = g2->nextGarbage();
         delete  g1 ;
      }
      delete m_garbageRoot;
   }

   if ( m_gstrRoot != 0 ) {
      GarbageString *g1 = m_gstrRoot;
      GarbageString *g2 = m_gstrRoot->nextGarbage();
      while( g2 != m_gstrRoot ) {
         g1 = g2;
         g2 = g2->nextGarbage();
         delete  g1 ;
      }
      delete m_gstrRoot;
   }

}


void MemPool::storeForGarbage( GarbageString *ptr )
{
   ptr->mark( currentMark() );

   if ( m_gstrRoot == 0 ) {
      m_gstrRoot = ptr;
      ptr->nextGarbage(ptr);
      ptr->prevGarbage(ptr);
   }
   else {
      ptr->prevGarbage( m_gstrRoot );
      ptr->nextGarbage( m_gstrRoot->nextGarbage() );
      m_gstrRoot->nextGarbage()->prevGarbage( ptr );
      m_gstrRoot->nextGarbage( ptr );
   }
   m_allocatedItems++;
   m_allocatedMem += ptr->allocated() + sizeof( *ptr );
}



void MemPool::destroyGarbage( GarbageString *ptr )
{
   ptr->nextGarbage()->prevGarbage( ptr->prevGarbage() );
   ptr->prevGarbage()->nextGarbage( ptr->nextGarbage() );
   m_allocatedItems--;
   m_allocatedMem -= (ptr->allocated() + sizeof( *ptr ) );
   if ( ptr == m_gstrRoot ) {
      m_gstrRoot = ptr->nextGarbage();
      // That was the last of the rings?
      if ( m_gstrRoot == ptr )
         m_gstrRoot = 0;
   }
   delete ptr;
}


void MemPool::storeForGarbage( Garbageable *ptr )
{
   ptr->mark( currentMark() );

   if ( m_garbageRoot == 0 ) {
      m_garbageRoot = ptr;
      ptr->nextGarbage(ptr);
      ptr->prevGarbage(ptr);
   }
   else {
      ptr->prevGarbage( m_garbageRoot );
      ptr->nextGarbage( m_garbageRoot->nextGarbage() );
      m_garbageRoot->nextGarbage()->prevGarbage( ptr );
      m_garbageRoot->nextGarbage( ptr );
   }
   m_allocatedItems++;
   m_allocatedMem += ptr->m_gcSize;
}

void MemPool::destroyGarbage( Garbageable *ptr )
{
   ptr->nextGarbage()->prevGarbage( ptr->prevGarbage() );
   ptr->prevGarbage()->nextGarbage( ptr->nextGarbage() );
   m_allocatedItems--;
   m_allocatedMem -= ptr->m_gcSize;
   if ( ptr == m_garbageRoot ) {
      m_garbageRoot = ptr->nextGarbage();
      // That was the last of the rings?
      if ( m_garbageRoot == ptr )
         m_garbageRoot = 0;
   }
   delete ptr;
}



GarbageLock *MemPool::lock( const Item &itm )
{
   GarbageLock *ptr = new GarbageLock( itm );

   // then add it in the availability pool
   if ( m_availPoolRoot == 0 ) {
      m_availPoolRoot = ptr;
      ptr->next(ptr);
      ptr->prev(ptr);
   }
   else {
      ptr->prev( m_availPoolRoot );
      ptr->next( m_availPoolRoot->next() );
      m_availPoolRoot->next()->prev( ptr );
      m_availPoolRoot->next( ptr );
   }

   return ptr;
}


void MemPool::unlock( GarbageLock *ptr )
{
   // frirst: remove the item from the availability pool
   ptr->next()->prev( ptr->prev() );
   ptr->prev()->next( ptr->next() );

   if ( ptr == m_availPoolRoot ) {
      m_availPoolRoot = ptr->next();
      // That was the last of the rings?
      if ( m_availPoolRoot == ptr )
         m_availPoolRoot = 0;
   }

   delete ptr;
}


bool MemPool::checkForGarbage()
{
   if ( m_autoClean ) {
      if ( m_allocatedMem > m_thresholdMemory )
      {
         performGC();
         return true;
      }
   }

   return false;
}

bool MemPool::gcMark()
{
   // first, invert mark bit.
   changeMark();

   m_aliveItems = 0;
   m_aliveMem = 0;


   // presume that all the registers need fresh marking
   markItemFast( m_owner->regA() );
   markItemFast( m_owner->regB() );
   markItemFast( m_owner->self() );
   markItemFast( m_owner->sender() );

   // mark the global symbols
   // When generational gc will be on, this won't be always needed.
   MapIterator iter = m_owner->liveModules().begin();
   while( iter.hasCurrent() )
   {
      LiveModule *currentMod = *(LiveModule **) iter.currentValue();
      // We must mark the current module.
      currentMod->mark( currentMark() );
      m_aliveMem += sizeof( LiveModule );
      m_aliveItems++;

      ItemVector *current = &currentMod->globals();
      for( uint32 j = 0; j < current->size(); j++ )
         markItemFast( current->itemAt( j ) );

      current = &currentMod->wkitems();
      for( uint32 k = 0; k < current->size(); k++ )
         markItemFast( current->itemAt( k ) );

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

      stack = ctx->getStack();
      for( pos = 0; pos < stack->size(); pos++ ) {
         // an invalid item marks the beginning of the call frame
         if ( stack->itemAt( pos ).type() == FLC_ITEM_INVALID )
            pos += VM_FRAME_SPACE - 1; // pos++
         else
            markItemFast( stack->itemAt( pos ) );
      }

      ctx_iter = ctx_iter->next();
   }

   // do the same for the locked pools
   if ( m_availPoolRoot != 0 )
   {
      GarbageLock *lock = m_availPoolRoot->next();
      while( lock != m_availPoolRoot ) {
         markItemFast( lock->item() );
         lock = lock->next();
      }
   }

   return true;
}

void MemPool::markItem( Item &item )
{
   switch( item.type() )
   {
      case FLC_ITEM_REFERENCE:
      {
         GarbageItem *gi = item.asReference();
         if( gi->mark() != currentMark() ) {
            m_aliveItems++;
            m_aliveMem += gi->m_gcSize;
            gi->mark( currentMark() );
            markItemFast( gi->origin() );
         }
      }
      break;

      case FLC_ITEM_LBIND:
         if ( item.asFBind() != 0 )
         {
            GarbageItem *gi = item.asFBind();
            if ( gi->mark() != currentMark() )
               gi->mark( currentMark() );
         }
         // fallback to string for the name part

      case FLC_ITEM_STRING:
         if ( item.asString()->garbageable() )
         {
            GarbageString *gs = static_cast< GarbageString *>( item.asString() );
            if ( gs->mark() != currentMark() )
            {
               gs->mark( currentMark() );
               m_aliveMem += item.asString()->allocated() + sizeof( GarbageString );
               m_aliveItems++;
            }
         }
      break;

      case FLC_ITEM_GCPTR:
         item.asGCPointerShell()->mark( currentMark() );
         break;

      case FLC_ITEM_ARRAY:
      {
         CoreArray *array = item.asArray();
         if( array->mark() != currentMark() ) {
            array->mark(currentMark());
            m_aliveItems++;
            m_aliveMem += array->m_gcSize;
            for( uint32 pos = 0; pos < array->length(); pos++ ) {
               markItemFast( array->at( pos ) );
            }

            // mark also the bindings
            if ( array->bindings() != 0 )
               array->bindings()->mark( currentMark() );

            // and also the table
            if ( array->table() != 0 )
               array->table()->mark( currentMark() );
         }
      }
      break;

      case FLC_ITEM_OBJECT:
      {
         CoreObject *co = item.asObject();
         if( co->mark() != currentMark() ) {
            co->mark( currentMark() );
            m_aliveItems++;
            m_aliveMem += co->m_gcSize;
            co->gcMarkData( currentMark() );
         }

      }
      break;

      case FLC_ITEM_FBOM:
      {
         // TODO: Optimize
         Item fbom;
         item.getFbomItem( fbom );
         markItemFast( fbom );
      }
      break;

      case FLC_ITEM_DICT:
      {
         CoreDict *cd = item.asDict();
         if( cd->mark() != currentMark() ) {
            cd->mark( currentMark() );
            m_aliveItems++;
            m_aliveMem += cd->m_gcSize;

            Item key, value;
            cd->traverseBegin();
            while( cd->traverseNext( key, value ) )
            {
               markItemFast( key );
               markItemFast( value );
            }
         }
      }
      break;

      case FLC_ITEM_METHOD:
      {
         // if the item isn't alive, give it the death blow.
         if ( item.asModule()->module() == 0 )
            item.setNil();
         else
         {
            CoreObject *co = item.asMethodObject();
            if( co->mark() != currentMark() ) {
               m_aliveItems++;
               m_aliveMem += co->m_gcSize;
               co->mark( currentMark() );
               co->gcMarkData( currentMark() );
            }

            // no need to mark the live modue;
            // if it's alive it has been marked by the main loop
         }
      }
      break;

      case FLC_ITEM_TABMETHOD:
      {
         // if the item isn't alive, give it the death blow.
         if ( item.asModule()->module() == 0 )
            item.setNil();
         else
         {
            Garbageable *co = item.asTabMethodArray();
            if ( co->mark() != currentMark() )
            {
               if( item.isTabMethodDict() )
               {
                  Item temp = item.asTabMethodDict();
                  markItemFast( temp );
               }
               else {
                  Item temp = item.asTabMethodArray();
                  markItemFast( temp );
               }
            }
            // no need to mark the live modue;
            // if it's alive it has been marked by the main loop
         }
      }
      break;

      case FLC_ITEM_CLSMETHOD:
      {
         CoreObject *co = item.asMethodObject();
         if( co->mark() != currentMark() ) {
            m_aliveItems++;
            m_aliveMem += co->m_gcSize;
            co->mark( currentMark() );
            // mark all the property values.
            co->gcMarkData( currentMark() );
         }

         CoreClass *cls = item.asMethodClass();
         if( cls->mark() != currentMark() ) {
            cls->mark( currentMark() );
            m_aliveItems++;
            m_aliveMem += cls->m_gcSize;
            markItemFast( cls->constructor() );
            for( uint32 i = 0; i <cls->properties().added(); i++ ) {
               markItemFast( *cls->properties().getValue(i) );
            }
         }
      }
      break;

      case FLC_ITEM_CLASS:
      {
         CoreClass *cls = item.asClass();
         if( cls->mark() != currentMark() ) {
            cls->mark( currentMark() );
            m_aliveItems++;
            m_aliveMem += cls->m_gcSize;
            markItemFast( cls->constructor() );
            for( uint32 i = 0; i <cls->properties().added(); i++ ) {
               markItemFast( *cls->properties().getValue(i) );
            }
         }
      }
      break;

      case FLC_ITEM_FUNC:
         // kill items referencing nothing
         if ( item.asModule()->module() == 0 )
            item.setNil();
      break;

      case FLC_ITEM_MEMBUF:
      {
         MemBuf *mb = item.asMemBuf();
         if ( mb->mark() != currentMark() )
         {
            m_aliveMem += mb->size() + sizeof( MemBuf );
            m_aliveItems++;

            mb->mark( currentMark() );
            CoreObject *co = item.asMemBuf()->dependant();
            // small optimization; resolve the problem here instead of looping again.
            if( co != 0 && co->mark() != currentMark() )
            {
               co->mark( currentMark() );
               m_aliveItems++;
               m_aliveMem += co->m_gcSize;
               co->gcMarkData( currentMark() );
            }
         }
      }
      break;

      // all the others are shallow items; already marked
   }

}


void MemPool::gcSweep()
{
   Garbageable *ring = ringRoot();
   if( ring != 0 )
   {
      Garbageable *ring2 = ring->nextGarbage();
      while( ring2 != ring ) {
         if ( ring2->mark() != currentMark() ) {
            ring2 = ring2->nextGarbage();
            destroyGarbage( ring2->prevGarbage() );
         }
         else
            ring2 = ring2->nextGarbage();
      }
      if ( ring->mark() != currentMark() )
         destroyGarbage( ring );

   }

   GarbageString *gsRing = m_gstrRoot;
   if( gsRing != 0 )
   {
      GarbageString *gsRing2 = gsRing->nextGarbage();
      while( gsRing2 != gsRing ) {
         if ( gsRing2->mark() != currentMark() ) {
            gsRing2 = gsRing2->nextGarbage();
            destroyGarbage( gsRing2->prevGarbage() );
         }
         else
            gsRing2 = gsRing2->nextGarbage();
      }
      if ( gsRing->mark() != currentMark() )
         destroyGarbage( gsRing );
   }
}

bool MemPool::performGC( bool bForceReclaim )
{
   m_aliveItems = 0;
   m_aliveMem = 0;

   // cannot perform?
   if ( ! gcMark() )
      return false;

   // is the memory enought to be reclaimed ?
   if ( bForceReclaim ||
        (m_allocatedMem - m_aliveMem ) > m_thresholdReclaim )
   {
      gcSweep();
      m_thresholdMemory = m_aliveMem +
                          (m_aliveMem / 3 ) + m_thresholdReclaim;
   }
   else {
      // it's useful to increase the threshold memory so that we
      // won't be called again too soon.
      m_thresholdMemory *= 2;
   }

   if ( m_thresholdMemory < m_setThreshold )
      m_thresholdMemory = m_setThreshold;

   return true;
}

}

/* end of mempool.cpp */

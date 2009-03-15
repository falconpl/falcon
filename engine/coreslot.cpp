/*
   FALCON - The Falcon Programming Language.
   FILE: coreslot.cpp

   Core Slot - Messaging slot system.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 11 Jan 2009 18:28:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coreslot.h>
#include <falcon/vm.h>
#include <falcon/traits.h>
#include <falcon/deepitem.h>
#include <falcon/vmmsg.h>


namespace Falcon {

bool coreslot_broadcast_internal( VMachine *vm )
{
   CoreIterator *ci = static_cast<CoreIterator *>( vm->local(0)->asGCPointer() );

   if ( ! ci->isValid() )
   {
      // were we called after a message?
      Item *msgItem = vm->local(4);
      if( msgItem->isInteger() )
      {
         VMMessage* msg = (VMMessage*) msgItem->asInteger();
         msg->onMsgComplete( true );
         delete msg;
      }
      
      // let the GC pointer to take care of the ietrator.
      return false;
   }
   
   Item current = ci->getCurrent();
   
   if ( ! current.isCallable() )
   {
      if( current.isComposed() )
      {
         // may throw
         current.asDeepItem()->readProperty( "on_" + *vm->local(3)->asString(), current );
      }
      else
      {
         throw new CodeError( ErrorParam( e_non_callable, __LINE__ ).extra( "broadcast" ) );
      }
   }
   
   int64 pfirst = vm->local(1)->asInteger();
   if( pfirst < 0 )
   {
      vm->pushParameter( *vm->local( 2 ) );
      vm->callFrame( current, 1 );
   }
   else 
   {
      int64 paramCount = vm->local(2)->asInteger();
      for( int32 i = 0; i < paramCount; i++ )
      {
         vm->pushParameter( *vm->param( (int32)(i + pfirst) ) );
      }
      vm->callFrame( current, (int32)paramCount );
   }
   
   ci->next();
   return true;
}


void CoreSlot::prepareBroadcast( VMachine *vm, uint32 pfirst, uint32 pcount, VMMessage *msg )
{
   CoreIterator *ci = getIterator();
   // nothing to broadcast?
   if( ! ci->isValid() )
   {
      delete ci;
      return;
   }

   vm->addLocals( 5 );
   vm->local(0)->setGCPointer( vm, ci );
   *vm->local(1) = (int64) pfirst;
   *vm->local(2) = (int64) pcount;
   *vm->local(3) = new CoreString( m_name );
   
   if ( msg != 0 )
   {
      // store it as an opaque pointer.
      vm->local(4)->setInteger( (int64) msg );
   }
   
   vm->returnHandler( &coreslot_broadcast_internal );
}


bool CoreSlot::remove( const Item &subscriber )
{
   CoreIterator *iter = getIterator();
   while( iter->isValid() )
   {
      if ( iter->getCurrent() == subscriber )
      {
         erase( iter );
         delete iter;
         return true;
      }

      iter->next();
   }

   delete iter;
   return false;
}

FalconData *CoreSlot::clone() const
{
   incref();
   return const_cast<CoreSlot*>(this);
}

void CoreSlot::gcMark( MemPool *mp ) 
{
   if ( m_bHasAssert )
      mp->markItem( m_assertion );
   
   ItemList::gcMark( mp );
}

void CoreSlot::assert( VMachine* vm, const Item &a )
{
   assert( a );
   if ( ! empty() )
   {
      vm->addLocals( 4 );
      vm->local(0)->setGCPointer( vm, getIterator() );
      *vm->local(1) = (int64) -1;
      *vm->local(2) = m_assertion;
      *vm->local(3) = new CoreString( m_name );
      
      vm->returnHandler( &coreslot_broadcast_internal );
   }
}


void CoreSlot::incref() const
{
   m_mtx.lock();
   m_refcount++;
   m_mtx.unlock();
}

void CoreSlot::decref()
{
   m_mtx.lock();
   bool bdel = --m_refcount == 0;
   m_mtx.unlock();
   
   if (bdel)
   {
      delete this;
   }
}


//=============================================================
// Traits for maps
// 
namespace traits
{
   CoreSlotPtrTraits &t_coreslotptr() { static CoreSlotPtrTraits dt; return dt; }
}


uint32 CoreSlotPtrTraits::memSize() const
{
   return sizeof( CoreSlot * );
}

void CoreSlotPtrTraits::init( void *itemZone ) const
{
   itemZone = 0;
}

void CoreSlotPtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   CoreSlot **target = (CoreSlot **) targetZone;
   CoreSlot *source = (CoreSlot *) sourceZone;

   *target = source;
}

int CoreSlotPtrTraits::compare( const void *firstz, const void *secondz ) const
{
   return ((int)firstz) - ((int)secondz);
}

void CoreSlotPtrTraits::destroy( void *item ) const
{
   CoreSlot *ptr = *(CoreSlot **) item;
   ptr->decref();
}

bool CoreSlotPtrTraits::owning() const
{
   return true;
}



}

/* end of coreslot.cpp */


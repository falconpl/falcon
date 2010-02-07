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
#include <falcon/iterator.h>

namespace Falcon {

bool coreslot_broadcast_internal( VMachine *vm )
{
   Iterator *ci = static_cast< Iterator *>( vm->local(0)->asGCPointer() );
   VMMessage* msg = 0;
   Item *msgItem = vm->local(4);
   if( msgItem->isInteger() )
   {
      msg = (VMMessage*) msgItem->asInteger();
   }

   if ( ! ci->hasCurrent() )
   {
      // were we called after a message?
      if( msg != 0 )
      {
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
      Item cache = *vm->local( 2 );
      vm->pushParam( cache );
      vm->callFrame( current, 1 );
   }
   else
   {
      int64 paramCount = 0;

      if( msg == 0 )
      {
         paramCount = vm->local(2)->asInteger();
         for( int32 i = 0; i < paramCount; i++ )
         {
            Item cache = *vm->param( (int32)(i + pfirst) );
            vm->pushParam( cache );
         }
      }
      else
      {
         paramCount = msg->paramCount();
         for ( uint32 i = 0; i < paramCount; ++i )
         {
            vm->pushParam( *msg->param(i) );
         }
      }

      vm->callFrame( current, (int32)paramCount );
   }

   ci->next();
   return true;
}


void CoreSlot::prepareBroadcast( VMContext *vmc, uint32 pfirst, uint32 pcount, VMMessage *msg )
{
   if( empty() )
   {
      return;
   }

   Iterator* iter = new Iterator( this );

   // and create a full functional frame.
   //vmc->createFrame( pcount );

   // we don't need to set the slot as owner, as we're sure it stays in range
   // (slots are marked) on themselves.
   vmc->addLocals( 5 );
   vmc->local(0)->setGCPointer( iter );
   *vmc->local(1) = (int64) pfirst;
   *vmc->local(2) = (int64) pcount;
   *vmc->local(3) = new CoreString( m_name );

   if ( msg != 0 )
   {
      // store it as an opaque pointer.
      vmc->local(4)->setInteger( (int64) msg );
   }

   vmc->returnHandler( &coreslot_broadcast_internal );
}


bool CoreSlot::remove( const Item &subscriber )
{
   Iterator iter( this );

   while( iter.hasCurrent() )
   {
      if ( iter.getCurrent() == subscriber )
      {
         erase( iter );
         return true;
      }

      iter.next();
   }

   return false;
}

CoreSlot *CoreSlot::clone() const
{
   incref();
   return const_cast<CoreSlot*>(this);
}

void CoreSlot::gcMark( uint32 mark )
{
   if ( m_bHasAssert )
      memPool->markItem( m_assertion );

   ItemList::gcMark( mark );
}

void CoreSlot::setAssertion( VMachine* vm, const Item &a )
{
   setAssertion( a );
   if ( ! empty() )
   {
      vm->addLocals( 5 ); // Warning -- we add 5 to nil the msg ptr callback at local(4).
      Iterator* iter = new Iterator( this );
      // we don't need to set the slot as owner, as we're sure it stays in range
      // (slots are marked) on themselves.
      vm->local(0)->setGCPointer( iter );
      *vm->local(1) = (int64) -1;
      *vm->local(2) = m_assertion;
      *vm->local(3) = new CoreString( m_name );
      // local(4) must stay nil; it's used by broadcast_internal as msg callback pointer.

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

void CoreSlot::getIterator( Iterator& tgt, bool tail ) const
{
   ItemList::getIterator( tgt, tail );
   const_cast<CoreSlot*>(this)->incref();
}
void CoreSlot::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   ItemList::copyIterator( tgt, source );
   const_cast<CoreSlot*>(this)->incref();
}

void CoreSlot::disposeIterator( Iterator& tgt ) const
{
   ItemList::disposeIterator( tgt );
   const_cast<CoreSlot*>(this)->decref();
}

//=============================================================
// Carrier for VM
//

CoreSlotCarrier::CoreSlotCarrier( const CoreClass* generator, CoreSlot* cs, bool bSeralizing ):
   FalconObject( generator, bSeralizing )
{
   if( cs != 0 )
   {
      cs->incref();
      m_user_data = cs;
   }
}

CoreSlotCarrier::CoreSlotCarrier( const CoreSlotCarrier &other ):
   FalconObject( other )
{
   // FalconObject takes care of cloning (increffing) the inner data.
}

CoreSlotCarrier::~CoreSlotCarrier()
{
    CoreSlot* cs = (CoreSlot*) m_user_data;
    cs->decref();
    // sterilize downward destructors
    m_user_data = 0;
}

void CoreSlotCarrier::setSlot( CoreSlot* cs )
{
   CoreSlot* old_cs = (CoreSlot*) m_user_data;
   if ( old_cs  != 0 )
      old_cs->decref();
   cs->incref();

   setUserData( cs );
}


CoreSlotCarrier *CoreSlotCarrier::clone() const
{
   return new CoreSlotCarrier( *this );
}

CoreObject* CoreSlotFactory( const CoreClass *cls, void *user_data, bool bDeserial )
{
   return new CoreSlotCarrier( cls, (CoreSlot *) user_data );
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
   if ( sizeof(int) == sizeof(void*))
      return (int)(((int64)firstz) - ((int64)secondz));
   else
      return (((int64)firstz) > ((int64)secondz)) ? -1 :
	     (((int64)firstz) < ((int64)secondz)) ? 1 : 0;
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


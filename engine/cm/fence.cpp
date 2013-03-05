/*
   FALCON - The Falcon Programming Language.
   FILE: fence.cpp

   Falcon core module -- Fence shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/cm/fence.cpp"

#include <falcon/classes/classshared.h>
#include <falcon/cm/fence.h>
#include <falcon/vmcontext.h>
#include <falcon/stdsteps.h>
#include <falcon/vm.h>

#include <falcon/errors/accesserror.h>
#include <falcon/errors/paramerror.h>
#include <falcon/stdhandlers.h>


namespace Falcon {
namespace Ext {

SharedFence::SharedFence( ContextManager* mgr, const Class* owner, int32 fenceCount, bool eventSemantic ):
         Shared(mgr, owner, 0 )
{
   m_fenceCount = fenceCount;
   m_level = fenceCount;
   m_bEventSemantic = eventSemantic;
}

SharedFence::~SharedFence()
{

}

void SharedFence::signal( int32 count )
{
   if( count <= 0 )
   {
      return;
   }

   lockSignals();
   uint32 oldLevel = m_level;
   m_level -= count;
   if( m_level <= 0 && oldLevel > 0)
   {
      Shared::lockedSignal(1);
   }
   unlockSignals();
}

int32 SharedFence::level() const
{
   lockSignals();
   int32 l = m_level;
   unlockSignals();
   return l;
}

int32 SharedFence::count() const
{
   lockSignals();
   int32 l = m_fenceCount;
   unlockSignals();
   return l;
}

void SharedFence::count( int32 count )
{
   lockSignals();
   m_fenceCount = count;
   unlockSignals();
}

int32 SharedFence::consumeSignal( VMContext* ctx, int32 )
{
   lockSignals();
   int32 result = Shared::lockedConsumeSignal(ctx, 1);
   if( result > 0 )
   {
      // yay, we can proceed!
      if( result )
      {
         if( m_bEventSemantic ) {
            m_level = m_fenceCount;
         }
         else {
            m_level += m_fenceCount;
         }
      }
   }
   unlockSignals();
   return result;
}

int32 SharedFence::lockedConsumeSignal( VMContext* ctx, int32)
{
   int32 result = Shared::lockedConsumeSignal(ctx, 1);
   if( result > 0 ) {
      // yay, we can proceed!
      if( result )
      {
         if( m_bEventSemantic ) {
            m_level = m_fenceCount;
         }
         else {
            m_level += m_fenceCount;
         }
      }
   }
   return result;
}

//=============================================================
//
//

/*#
  @property level Fence
  @brief Current signal level of the fence.

  This is a read/only property.
 */
static void get_level( const Class*, const String&, void* instance, Item& value )
{
   SharedFence* shared = static_cast<SharedFence*>(instance);
   value.setInteger(shared->level());
}


/*#
  @property isEvent Fence
  @brief True if the current fence has event semantic

  This is a read/only property.
 */
static void get_isEvent( const Class*, const String&, void* instance, Item& value )
{
   SharedFence* shared = static_cast<SharedFence*>(instance);
   value.setBoolean(shared->isEvent());
}

/*#
  @property count Fence
  @brief Count of signals that this fence waits on.

  This is property can be changed at runtime, but the
  count level doesn't affect the level. In other words,
  the count change becomes effective after
  the first waiter has waken up.

  For example,
  if a fence with count 5 received 2 signals, and it is
  now at level 3, setting count to 2 won't cause the
  waiters to wake up; the fence still need to receive
  3 signals before waiters are waken up.
 */
static void get_count( const Class*, const String&, void* instance, Item& value )

{
   SharedFence* shared = static_cast<SharedFence*>(instance);
   value.setInteger(shared->count());
}


static void set_count( const Class*, const String&, void* instance, const Item& value )

{
   if(!value.isOrdinal())
   {
      throw FALCON_SIGN_XERROR( AccessError, e_inv_prop_value, .extra("N > 0") );
   }

   int32 count = (int32) value.forceInteger();
   if( count <= 0 )
   {
      throw FALCON_SIGN_XERROR( AccessError, e_inv_prop_value, .extra("N > 0") );
   }

   SharedFence* shared = static_cast<SharedFence*>(instance);
   shared->count( count );
}


/*#
  @method signal Fence
  @brief Signals the fence
  @optparam count Count of signals to be sent to the fence.

   The parameter @b count must be greater or equal to 1.
 */
FALCON_DECLARE_FUNCTION( signal, "count:[N]" );
void Function_signal::invoke( VMContext* ctx, int32 pCount )
{
   SharedFence* evt = static_cast<SharedFence*>(ctx->self().asInst());
   int32 count = pCount > 0 ? ((int32)ctx->param(0)->forceInteger()) : 1;
   if( count <= 0 )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("[N]>0") );
   }

   evt->signal( count );
   ctx->returnFrame();
}


ClassFence::ClassFence():
   ClassShared("Fence") 
{
   static Class* shared = Engine::handlers()->sharedClass();
   setParent(shared);

   addProperty( "level", &get_level );
   addProperty( "isEvent", &get_isEvent );
   addProperty( "count", &get_count, &set_count );
   
   addMethod( new Function_signal );
}


ClassFence::~ClassFence()
{}

void* ClassFence::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassFence::op_init( VMContext* ctx, void*, int pcount ) const
{
   int32 count;
   bool isEvent = false;

   Item* params = ctx->opcodeParams(pcount);

   if( pcount < 1 || ! params[0].isOrdinal() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("N>0,[B]") );
   }

   count = (int32) params[0].forceInteger();
   if( count <= 0 )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_range, .extra("N>0,[B]") );
   }

   if( pcount >= 2 )
   {
      isEvent = params[1].isOrdinal();
   }

   SharedFence* sb = new SharedFence(&ctx->vm()->contextManager(), this, count, isEvent);
   ctx->stackResult(pcount+1, FALCON_GC_STORE(this, sb));
   return true;
}

}
}

/* end of semaphore.cpp */

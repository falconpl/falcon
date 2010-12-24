/*
   FALCON - The Falcon Programming Language.
   FILE: gc_ext.cpp

   Garbage control from scripts
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 02:10:57 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/memory.h>

/*#
   @beginmodule core
*/

namespace Falcon {
namespace core {

/*#
   @page gc_control About the garbage collector.
   
   The standard collector strategy (be it set up by the Falcon interpreter or
   by embedding applications) is adequate for average scripts.

   However, some script meant to start from command line and dealing with time
   critical data may find the action of the garbage collector too intrusive. For example
   the GC may occuur at the wrong time. Other times, calculation intensive programs
   may generate a lot of data that they know in advance can be never garbaged
   during some period. In those case, having GC to scan periodically the
   allocated memory for released blocks is evidently a useless waste of time.

   Finally, some complex scripts may even provide their own collection strategy,
   based on memory pattern usage that they know in advance. Starting the collection
   loop at time intervals, provided the memory allocation has grown at a certain rate,
   or hasn't grown for a certain time, may be a fitting strategy for some scripts.

   A sensible usage of the garbage collection feature may boost performance of
   calculation and memory intensive scripts by order of degrees, and may be
   essential in time critical applications where some part of the process has to
   be performed as fast as possible.

   Consider that some of the functions listed in this section may not be always available.
   Some embedding application may decide to turn some or all of them off for security
   reasons, as a malevolent script may crash an application very fast by turning
   off automatic GC check-and-reclaim feature and then creating a great amount
   of garbage. Also, the loop maximum execution time control is not present by
   default in Falcon command line, as the time-deterministic version of the
   garbage collector is sensibly slower, and it would be useless to the
   vast majority of the scripts.
*/

/*# @object GC
   @brief Support for script-based garbage collection strategies.
   @prop usedMem Memory used by the Falcon engine.
   @prop items Single GC sensible items currently allocated.
   @prop th_normal Threshold of occupied memory above which the GC will enter the normal mode.
   @prop th_active Threshold of occupied memory above which the GC will enter the active mode.

   @see gc_control
*/

CoreObject* GC_Factory( const CoreClass *cls, void *user_data, bool )
{
   // just to mark our user data.
   return new ReflectObject( cls, (void*)1 );
}

/*#
   @method enable GC
   @brief Turns automatic GC feature on or off.
   @param mode true to turn automatic GC on, false to turn it off.

   Virtual machines and some heavy garbage generating functions call
   periodically a function that checks for the level of allocated
   memory to have reached a critical point. When there is too much allocated
   memory of uncertain status, a garbage collecting loop is started.

   By setting this property to false, this automatic control is skipped, and allocated
   memory can grow up to physical process limits (or VM memory limit
   constraints, if provided). Setting this value to true will cause VM to
   perform memory control checks again with the usual strategy.

   In case the script is sure to have generated a wide amount of garbage, it
   is advisable to call explicitly @a GC.perform() before turning automatic GC on,
   as the "natural" collection loop may start at any later moment, also after
   several VM loops.
*/

FALCON_FUNC  GC_enable( ::Falcon::VMachine *vm )
{
   if( vm->param(0) == 0 )
      vm->retval( vm->isGcEnabled() ? 1 : 0 );
   else
      vm->gcEnable(vm->param(0)->isTrue());
}


/*#
   @method perform GC
   @brief Requests immediate check of garbage.
   @optparam wcoll Set to true to wait for the collection of free memory to be complete.
   @return true if the gc has been actually performed, false otherwise.

   Suspends the activity of the calling Virtual Machine, waiting for the
   garbage collector to complete a scan loop before proceeding.
*/

FALCON_FUNC  GC_perform( ::Falcon::VMachine *vm )
{
   if ( vm->paramCount() > 0 )
      vm->performGC( vm->param(0)->isTrue() );
   else
      vm->performGC();
}

/*#
   @method adjust GC
   @brief Sets or gets the automatic threshold levels adjust algorithm.
   @optparam mode The adjust mode used by the GC.
   @return The mode currently set.

   Mode can be one of:
   - GC.ADJ_NONE: No adjust. All adjusting must be done manually.
   - GC.ADJ_STRICT: Aggressive adjustment strategy, forcing active collection whenever
                    the memory grows.
   - GC.ADJ_LOOSE: Permissive adjustment strategy, forcing active collection only
                   when memory grows promptly.
   - GC.ADJ_SMOOTH_FAST: Adjustment following the memory allocation status with some
                         delay and a smooth asymptotic curve (fast adaption).
   - GC.ADJ_SMOOTH_SLOW: Adjustment following the memory allocation status with some
                         delay and a smooth asymptotic curve (slow adaption).

*/

FALCON_FUNC  GC_adjust( ::Falcon::VMachine *vm )
{
   Item *i_setting = vm->param(0);
   vm->retval( memPool->rampMode() );

   if ( i_setting != 0 )
   {
      if ( ! i_setting->isOrdinal() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin(e_orig_runtime)
            .extra( "N" ) );
      }
      else if ( ! memPool->rampMode( (int) i_setting->forceInteger() ) )
      {
         throw new ParamError( ErrorParam( e_param_range, __LINE__ )
            .origin(e_orig_runtime) );
      }
   }
}

// Reflective path method
void GC_usedMem_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) gcMemAllocated();
}

void GC_aliveMem_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) gcMemAllocated();
}

void GC_items_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) memPool->allocatedItems();
}

void GC_th_normal_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) memPool->thresholdNormal();
}

void GC_th_normal_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   if ( ! property.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "N" ) );
   }

   memPool->thresholdActive( (size_t) property.forceInteger() );
}


void GC_th_active_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) memPool->thresholdActive();
}


void GC_th_active_rto(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{

   if ( ! property.isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "[N]" ) );
   }

   memPool->thresholdActive( (size_t) property.forceInteger() );
}

}
}

/* end of gc_ext.cpp */

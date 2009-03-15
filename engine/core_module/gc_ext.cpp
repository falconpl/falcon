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

*/
namespace Falcon {
namespace core {

/*#
   @funset gc_control Garbage collecting control
   @brief Support for script-based garbage collection strategies.

   The standard collector strategy (be it set up by the Falcon interpreter or
   by embedding applications) is adequate for average scripts.

   However, some script meant to start from command line and dealing with time
   critical data may find the action of the garbage collector too intrusive. In example
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
   @prop usedMem Memory used by the Falcon engine
   @prop aliveMem Memory found alive in last scans (not implemented)
   @prop items Single GC sensible items accounted (not implemented).
   @prop th_normal Threshold of occupied memory above which the GC will enter the normal mode.
   @prop th_active Threshold of occupied memory above which the GC will enter the active mode.
   
*/

FALCON_FUNC  GC_init( ::Falcon::VMachine *vm )
{
   // just to mark our user data.
   
   vm->self().asObject()->setUserData( (void*) 1 );
}

/*#
   @method enable GC
   @inset gc_control
   @brief Turns automatic GC feature on or off.
   @param mode true to turn automatic GC on, false to turn it off.

   Virtual machines and some heavy garbage generating functions call
   periodically a function that checks for the level of allocated
   memory to have reached a critical point. When there is too much allocated
   memory of uncertain status, a garbage collecting loop is started.

   By setting gcEnable to off, this automatic control is skipped, and allocated
   memory can grow up to physical process limits (or VM memory limit
   constraints, if provided). Setting this value to true will cause VM to
   perform memory control checks again with the usual strategy.

   In case the script is sure to have generated a wide amount of garbage, it
   is advisable to call explicitly gcPerform() before turning automatic GC on,
   as the “natural” collection loop may start at any later moment, also after
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
   @inset gc_control
   @brief Requests immediate check of garbage.
   @optparam bForce If true, force collection of unused memory.
   @return true if the gc has been actually performed, false otherwise.

   Performs immediately a garbage collection loop. All the items the script
   is managing are checked, and the memory they occupy is stored for
   later reference; the findings of the perform loop may be retrieved
   by the gcGetParams function. If the memory that is found unreferenced
   is below the memory reclaim threshold level, the function returns immediately false.
   Otherwise, a reclaim loop is performed and some memory gets cleared,
   and the function returns true.

   If @b bForce is set to true, all the unused memory is immediately reclaimed,
   without taking into consideration the reclaim threshold.

   See @a gcSetThreshold function for a in-depth of thresholds and memory
   constraints.
*/

FALCON_FUNC  GC_perform( ::Falcon::VMachine *vm )
{
   bool bRec;
   /*
   if ( vm->param( 0 ) != 0 )
   {
      bRec = vm->param( 0 )->isTrue();
   }
   else {
      bRec = false;
   }

   vm->retval( memPool->performGC( bRec ) ? 1 : 0 );
   */
}

// Reflective path method
void GC_usedMem_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property =  (int64) gcMemAllocated();
}

void GC_aliveMem_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   property = (int64) gcMemAllocated();
}

void GC_items_rfrom(CoreObject *instance, void *user_data, Item &property, const PropEntry& )
{
   //TODO
   property = (int64) 0;
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

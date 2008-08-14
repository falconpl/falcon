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

/*#
   @beginmodule core
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

/*#
   @function gcEnable
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

FALCON_FUNC  gcEnable( ::Falcon::VMachine *vm )
{
   if( vm->param(0) == 0 )
      vm->retval( vm->memPool()->autoCleanMode() ? 1 : 0 );
   else
      vm->memPool()->autoCleanMode( vm->param(0)->isTrue() );
}

/*#
   @function gcSetThreshold
   @inset gc_control
   @brief Turns automatic GC feature on or off.
   @optparam scanTh Amount of memory (in Kb) that triggers garbage scan.
   @optparam collectTh Amount of memory (in Kb) that triggers garbage collection.

   This function sets the garbage collection threshold levels that start automatic
   GC loop. They are both optional; if nil is provided in place of one of them,
   that value is ignored (and stays at it was before).

   The scanTh parameter determines when the automatic check the VM performs triggers
   a collection loop. When the allocated memory is below the scanTh level,
   nothing happens; when it is above, a collection loop is started.

   When the VM determines how much memory is garbage, it checks that value against
   the collectTh level. A reclaim loop is started only if the detected free memory
   is more than collectTh bytes.

   While scanTh value is not used if @a gcEnable is turned to off, collectTh
   level will still determine if the claim loop is worth to be taken also in case
   of explicit @a gcPerform calls from scripts.

   The GC level does not take into consideration the real amount of memory that the
   objects are using, but the memory they report to the VM when they are created or
   modified.

   After a GC is performed, the threshold levels are automatically adjusted so that GC
   collection checks and reclaims are not performed too often. The adjustment may
   change on a per VM basis; the default is to set the scanTh to the double of the
   memory that was found actually used, without changing the collectTh.

   Default scanTh is very strict (1MB in this version); this level has the advantage
   to scale up fast in case of huge scripts while still being big enough to let the
   vast majority of control/embedded scripts to run without GC hindrance for their
   whole life.
*/

FALCON_FUNC  gcSetThreshold( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param( 0 );
   Item *p1 = vm->param( 1 );
   bool done = false;

   if( p0 != 0 && p0->isOrdinal() ) {
      done = true;
      vm->memPool()->thresholdMemory( (uint32) p0->forceInteger() );
   }

   if( p1 != 0 && p1->isOrdinal() ) {
      done = true;
      vm->memPool()->reclaimLevel( (uint32) p1->forceInteger() );
   }

   if ( ! done )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( [N], [N] )" ) ) );
   }
}

/*#
   @function gcSetTimeout
   @inset gc_control
   @brief Turns automatic GC feature on or off.
   @param msTimeout Amount of memory (in Kb) that triggers garbage scan.

   This function sets the maximum time a collection loop may take, expressed
   in milliseconds. A timeout of 0 means "infinite".

   This functionality is turned off by default. It must be explicitly activated
   by requesting it at the Falcon command line interpreter, or in case of
   embedding applications, it must be explicitly provided to the VM that
   is hosting the script.

   This is for two reasons: mainly, checking for time constraints is itself
   a time consuming operation, especially when compared to the atomic operations
   that a GC loop usually performs. Secondly, a script using this feature
   may crash the host application very soon once environmental condition changes.

   Considering the fact that without a complete GC loop used memory will never
   decrease, it's easy to see that a strict time constraint may prevent a full
   GC loop to ever take place. Even if the time is wide enough under normal
   circumstances (i.e. 100ms), if the script is to run ONCE on a heavy used
   CPU it may end up to be unable to perform the GC loop in the specified
   time, causing the next loops to be penalized, and possibly to never
   be able again to complete the collection under the given constraints.

   If a script and the hosting program are willing to use this feature, the
   time constraints must be used only when it is really needed; they must
   be periodically turned off (by setting the timeout to 0) or possibly not used at all.

   Of course, this is valid only for the time-constraint aware default GC provided by
   the Falcon API. Embedders are free to implement progressive GCs that may
   actually perform partial scanning in given time constraints. The "three-colors"
   garbage collector is a classical example. As all the GC strategies providing
   this feature are intrinsically less efficient than a monolithic all-or-nothing
   approach, and as they turn to be useful only in a very small category of
   applications (usually not delegated to scripting languages), Falcon does not
   provide any of them (for now).

   However, as the embedders may find this hook useful, it is provided here in
   the core module, where it interfaces directly with a property on the base
   class handling memory and garbage: MemPool.
*/

FALCON_FUNC  gcSetTimeout( ::Falcon::VMachine *vm )
{
   Item *p0 = vm->param( 0 );
   bool done = false;

   if( p0 != 0 && p0->isOrdinal() ) {
      vm->memPool()->setTimeout( (uint32) p0->forceInteger() );
   }
   else
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( N )" ) ) );
   }
}

/*#
   @function gcPerform
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

FALCON_FUNC  gcPerform( ::Falcon::VMachine *vm )
{
   bool bRec;

   if ( vm->param( 0 ) != 0 )
   {
      bRec = vm->param( 0 )->isTrue();
   }
   else {
      bRec = false;
   }

   vm->retval( vm->memPool()->performGC( bRec ) ? 1 : 0 );
}

/*#
   @function gcGetParams
   @inset gc_control
   @brief Requests immediate check of garbage.
   @optparam amem Amount of allocated memory.
   @optparam aitm Amount of allocated items.
   @optparam lmem Amount of alive memory.
   @optparam litm Amount of alive items.
   @optparam sth Currently set scan threshold.
   @optparam cth Currently set collect threshold.
   @optparam to Currently set collection timeout.

   This function retreives several statistical data and settings from the
   virtual machine memory manager. To retreive a parameter, pass a variable
   by reference, and pass nil in place of uninteresting parameters; in example,
   if wishing to recover the amount of alive memory and items:
   @code
   aliveItems = 0
   aliveMemory = 0
   gcGetParams( nil, nil, $aliveMemory, $aliveItems )
   @endcode

   Extra unneeded informations may just be ignored.

   The value of the alive memory and items is reliable only after (actually, soon after)
   a gcPerform() call. The data is also stored by automatic GC loops invoked
   by the VM, but the scripts have no mean to know when they are performed.
   It is reasonable to suppose that the information is still not available if
   the values returned are both 0.

   Both alive and allocated item count refers to those items that are actually stored
   in the memory manager. In fact, not every falcon item being available to the program
   will require garbage collection; only complex and deep items are accounted
   (mainly objects, strings, arrays and dictionaries).

   The informations returned by this function may be used by scripts both for
   debugging purpose and to build their own collection strategy.
*/

FALCON_FUNC  gcGetParams( ::Falcon::VMachine *vm )
{
   Item *i_mpAllocMem = vm->param( 0 );
   Item *i_mpAllocItems = vm->param( 1 );
   Item *i_mpAliveMem = vm->param( 2 );
   Item *i_mpAliveItems = vm->param( 3 );
   Item *i_mpThreshold = vm->param( 4 );
   Item *i_mpRecLev = vm->param( 5 );
   Item *i_mpTimeout = vm->param( 6 );

   if( i_mpAllocMem != 0 )
      i_mpAllocMem->setInteger( vm->memPool()->allocatedMem() );
   if( i_mpAllocItems != 0 )
      i_mpAllocItems->setInteger( vm->memPool()->allocatedItems() );
   if( i_mpAliveMem != 0 )
      i_mpAliveMem->setInteger( vm->memPool()->aliveMem() );
   if( i_mpAliveItems != 0 )
      i_mpAliveItems->setInteger( vm->memPool()->aliveItems() );
   if( i_mpThreshold != 0 )
      i_mpThreshold->setInteger( vm->memPool()->thresholdMemory() );
   if( i_mpRecLev != 0 )
      i_mpRecLev->setInteger( vm->memPool()->reclaimLevel() );
   if( i_mpTimeout != 0 )
      i_mpTimeout->setInteger( vm->memPool()->getTimeout() );
}

}
}

/* end of gc_ext.cpp */

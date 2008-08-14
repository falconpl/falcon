/*
   FALCON - The Falcon Programming Language.
   FILE: coroutine_ext.cpp

   Coroutine support, sleep functions, kind request functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include "../vmsema.h"

/*#
   @beginmodule core
*/

namespace Falcon {
namespace core {

/*@begingroup coro_sup Coroutine support

   The functions in this group allows to interact with the coroutine support that is
   provided by the Virtual Machine. Most of them translate in requests to the virtual
   machine.

   Also, functions in this group are meant to permeate the embedding applications.
   Requests generate by coroutining are delivered to the application for approval
   and control.
*/

/*#
   @function yield
   @ingroup coroutine_support
   @brief gives up the rest of the coroutine time slice.

   This signals the VM that the current coroutine is temporarily done, and that another
   coroutine may be executed. If no other coroutines can be executed, current coroutine
   is resumed immediately (actually, it is never swapped out).
*/
FALCON_FUNC  yield ( ::Falcon::VMachine *vm )
{
   vm->yieldRequest( 0.0 );
}

/*#
   @function yieldOut
   @brief Terminates current coroutine.
   @ingroup coroutine_support
   @param retval a return value for the coroutine.

   The current coroutine is terminated. If this is the last coroutine,
   the VM exits. Calling this function has the same effect of the END
   virtual machine PCODE.

   In multithreading context, exiting from the last coroutine causes
   a clean termination of the current thread.

   @see exit
*/
FALCON_FUNC  yieldOut ( ::Falcon::VMachine *vm )
{
   Item *ret = vm->param(0);
   vm->yieldRequest( -1.0 );
   if ( ret != 0 )
      vm->retval( *ret );
   else
      vm->retnil();
}


/*#
   @function sleep
   @brief Put the current coroutine at sleep for some time.
   @param time Time, in seconds and fractions, that the coroutine wishes to sleep.
   @return an item posted by the embedding application.
   @ingroup coroutine_support
   @raise InterruptedError in case of asynchronous interruption.

   This function declares that the current coroutines is not willing to proceed at
   least for the given time. The VM will swap out the coroutine until the time has
   elapsed, and will make it eligible to run again after the given time lapse.

   The parameter may be a floating point number if a pause shorter than a second is
   required.

   The @b sleep() function can be called also when the VM has not started any coroutine;
   this will make it to be idle for the required time.

   In embedding applications, the Virtual Machine can be instructed to detect needed idle
   time and return to the calling application instead of performing a system request to
   sleep. In this way, embedding applications can use the idle time of the virtual machine
   to perform background operations. Single threaded applications may continue their execution
   and schedule continuation of the Virtual Machine at a later time, and multi-threaded
   applications can perform background message processing.

   This function complies with the interrupt protocol. The Virtual Machine may be
   asynchronously interrupted from the outside (i.e. from another thread). In this case,
   @b sleep will immediately raise an @a InterruptedError instance. The script may just
   ignore this exception and let the VM to terminate immediately, or it may honor the
   request after a cleanup it provides in a @b catch block, or it may simply ignore the
   request and continue the execution by discarding the error through an appropriate
   @b catch block.

   @see interrupt_protocol
*/

FALCON_FUNC  _f_sleep ( ::Falcon::VMachine *vm )
{
   Item *amount = vm->param(0);
   numeric pause;
   if( amount == 0 )
      pause = 0.0;
   else {
      pause = amount->forceNumeric();
      if ( pause < 0.0 )
         pause = 0.0;
   }

   vm->yieldRequest( pause );
}

/*#
   @function beginCritical
   @brief Signals the VM that this coroutine must not be interrupted.
   @ingroup coroutine_support

   After this call the VM will abstain from swapping this coroutine out
   of the execution context. The coroutine can then alter a set of data that
   must be prepare and readied for other coroutines, and then call @a endCritical
   or @a yield to pass the control back to the other coroutines.

   This function is not recursive. Successive calls to @b beginCritical are not
   counted, and have actually no effect. The first call to @a yield will swap
   out the coroutine, and the first call to @a endCritical will signal the
   availability of the routine to be swapped out, no matter how many
   times @a beginCritical has been called.
*/

FALCON_FUNC  beginCritical ( ::Falcon::VMachine *vm )
{
   vm->allowYield( false );
}

/*#
   @function endCritical
   @brief Signals the VM that this coroutine can be interrupted.
   @ingroup coroutine_support

   After this call, the coroutine may be swapped. This will happen
   only if/when the timeslice for this coroutine is over.

   This function is not recursive. Successive calls to @a beginCritical
   are not counted, and have actually no effect.
   The first call to @a yield will swap out the coroutine, and the first
   call to @a endCritical will signal the availability of the routine to be
   swapped out, no matter how many times @a beginCritical has been called.
*/

FALCON_FUNC  endCritical ( ::Falcon::VMachine *vm )
{
   vm->allowYield( true );
}

/*#
   @class Semaphore
   @brief Simple coroutine synchronization device.
   @ingroup coroutine_support
   @optparam initValue Initial value for the semaphore; if not given, 0 will be assumed.

   The semaphore is a simple synchronization object that is used by coroutines to
   communicate each others about relevant changes in the status of the application.

   Decrements the value of the semaphore, and eventually waits for the value to be > 0.
   When a @a Semaphore.wait method is called on a semaphore, two things may happen:
   if the value of the semaphore is greater than zero, the value is decremented and the
   coroutine can proceed. If it's zero, the coroutine is swapped out until the
   semaphore gets greater than zero again. When this happens, the coroutine
   decrements the value of the semaphore and proceeds. If a timeout parameter is given,
   in case the semaphore wasn't posted before the given timeout the function will return
   false.

   The order by which coroutines are resumed is the same by which they asked
   to wait on a semaphore. In this sense, @a Semaphore.wait method is implemented as a fair
   wait routine.

   The @a Semaphore.post method will raise the count of the semaphore by the given parameter
   (1 is the default if the parameter is not given). However, the calling coroutine
   won't necessarily be swapped out until a @a yield is called.
*/

/*#
   @init Semaphore
   @brief Initializes the semaphore.

   By default, the semaphore is initialized to zero; this means that the
   first wait will block the waiting coroutine, unless a @a Semaphore.post
   is issued first.
*/
FALCON_FUNC  Semaphore_init ( ::Falcon::VMachine *vm )
{
   Item *qty = vm->param(0);
   int32 value = 0;
   if ( qty != 0 ) {
      if ( qty->type() == FLC_ITEM_INT )
         value = (int32) qty->asInteger();
      else if ( qty->type() == FLC_ITEM_NUM )
         value = (int32) qty->asNumeric();
      else {
         vm->raiseRTError( new ParamError( ErrorParam( e_param_outside ).extra( "( N )" ) ) );
         return;
      }
   }

   VMSemaphore *sem = new VMSemaphore( value );
   vm->self().asObject()->setUserData( sem );
}

/*#
   @method post Semaphore
   @brief Increments the count of the semaphore.
   @optparam count The amount by which the semaphore will be incremented (1 by default).

   This method will increment the count of the semaphore by 1 or by a specified amount,
   allowing the same number of waiting coroutines to proceed.

   However, the calling coroutine won't necessarily be swapped out until a @a yield is called.
*/
FALCON_FUNC  Semaphore_post ( ::Falcon::VMachine *vm )
{
   VMSemaphore *semaphore = static_cast< VMSemaphore *>(vm->self().asObject()->getUserData());
   Item *qty = vm->param(0);
   int32 value = 1;
   if ( qty != 0 ) {
      if ( qty->type() == FLC_ITEM_INT )
         value = (int32)qty->asInteger();
      else if ( qty->type() == FLC_ITEM_NUM )
         value = (int32) qty->asNumeric();
      else {
         vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( N )" ) ) );
         return;
      }
      if (value <= 0)
         value = 1;
   }

   semaphore->post( vm, value );
}

/*#
   @method wait Semaphore
   @brief Waits on a semaphore.
   @optparam timeout Optional maximum wait in seconds.

   Decrements the value of the semaphore, and eventually waits for the value to be greater
   than zero.
*/
FALCON_FUNC  Semaphore_wait ( ::Falcon::VMachine *vm )
{
   VMSemaphore *semaphore = static_cast< VMSemaphore *>(vm->self().asObject()->getUserData());
   Item *i_wc = vm->param( 0 );
   if ( i_wc == 0 )
      semaphore->wait( vm );
   else {
      if ( ! i_wc->isOrdinal() )
	  {
	     vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "( N )" ) ) );
         return;
	  }
	  semaphore->wait( vm, i_wc->forceNumeric() );
   }

}

/*#
   @function suspend
   @brief Temporarily suspends the execution of the Virtual Machine.
   @optparam timeout Optional maximum wait in seconds.
   @return an item posted by the embedding application.
   @ingroup coroutine_support

   This function is meant to provide a very simple means of communication
   with the embedding application. Other means are provided as well; they all
   require less work on the script side, but more integration work on the embedding
   application side. So, this communcation mean may be interesting for very simple
   embedding applications, or in the early stage of the integration process.

   This function hasn't any utility for stand-alone applications, and should not
   be used by them, except than for testing.

   After this call, the VM exits immediately and the control is given back to the
   embedder. The embedder can check for the VM to have exited because of a suspend
   call, and in this case, it may call the VM method Falcon::VMachine::resume()
   to make the script to proceed from the point it was suspended.

   The resume method of the virtual machine may accept an item that is then passed
   to the suspended script as the return value of the suspend call. In this way,
   the script may receive a callback notification of events happening in the main
   application, and manage them. This is an example:

   @code
   lastEvent = “none”

   while lastEvent != “quit”

      lastEvent = suspend()

      switch lastEvent
         case “this”
            doThis()
         case “that”
            doThat()
      end
   end
   @endcode

   The kind of item that is returned by suspend() call is a convention between the
   script and the embedding application, and may be a complex item as well as an
   instance of an embedder specific class.

   While in suspended state, the VM may be also destroyed, or the calling module may
   be unlinked from it, or the execution may be restarted instead of resumed.
*/

FALCON_FUNC vmSuspend( ::Falcon::VMachine *vm )
{
   vm->requestSuspend();
}

}
}

/* end of coroutine_ext.cpp */

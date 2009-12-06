/*
   FALCON - The Falcon Programming Language.
   FILE: continuation.h

   Continuation object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Dec 2009 17:04:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/continuation.h>
#include <falcon/mempool.h>
#include <falcon/globals.h>

namespace Falcon {
namespace core {


//===========================================================
//

/*#
   @class Continuation
   @brief Intra-context suspension and resume control device.
   @param callable A function or any callable item.

   This instances of this class can be used to execute some
   code that can be interrupted at any time and resumed later
   from the point they were interrupted.

   The function or other callable item set as the parameter
   in the constructor of this class will receive a special
   copy of the instance when this instance is called
   as a @i functor. For example:
   @code
      function callable( cont )
         > cont.toString()
      end

      c = Continuation( callable )
      c()                         // will call callable( c )
   @endcode

   The special instance which is passed to the @b callable item is itself
   a functor. Calling it causes immediate suspension and return to the
   original frame where the continuation instance was first called. An optional
   value can be returned to the caller by passing it as a parameter of the
   continuation instance.

   For example; the following code returns to the continuation first caller
   if the random number matches a dividend of a "secret" number.

   @code
      c = Continuation( { secret, c =>
            while true
               r = random( 2, secret )
               if secret % r == 0
                  c(r)                 // return "r" to our caller
               end
            end })

      > "A random factor of 136: ", c(136)
   @endcode

   Other than returning immediately to the first caller of the continuation,
   the current state of the called sequence is recorded and restored when
   subsequent calls are performed. In those calls, parameters are ignored
   (they stay the same as the first call). The following code returns the
   position where a given element is found in an array:

   @code
      c = Continuation( { elem, array, c =>
            for n in [0: array.len()]
               if array[n] == elem: c(n)
            end })

      while (pos = c(10, [1,"a",10,5,10] ))
         > "Found a '10' at pos ", pos
      end
   @endcode

   @note It is not possible to use continuations in atomic calls; for/in
         generators are called as atomically, so it's not possible to use
         continuations as generators in for/in loops. Use standard \b while
         loops instead.

   Separate continuations calling the same functions have a completely different
   state.

   Also, the @a Continuation.reset method clears any previously existing state of a
   continuation, so that it can be called anew again without the need to recreate it.

 */
FALCON_FUNC Continuation_init ( ::Falcon::VMachine *vm )
{
   Item* i_cc = vm->param(0);
   if ( i_cc == 0 || ! i_cc->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
            extra("C") );
   }

   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   cc->cont( new Continuation(vm) );
   cc->ccItem( *i_cc );
   cc->getMethod("_suspend",  cc->suspendItem() );
}

/*#
    @method __call Continuation
    @brief Enters into or exit from a continuation.
    @param ... parameters passed to the callable stored in the instance, or to the return value.
    @return The parameter passed to this method from inside the continuation.

    For the complete usage pattern, see the @a Continuation class.
 */
FALCON_FUNC Continuation_call ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   // call in passive phase calls the desired item.
   if( cc->cont()->jump() )
      return;

   for( int32 i = 0; i < vm->paramCount(); i++)
   {
      vm->pushParam( *vm->param(i) );
   }
   // otherwise, we have to call the item from here.
   vm->pushParam( cc->suspendItem() );
   vm->callFrame( cc->ccItem(), 1 + vm->paramCount() );
}

/*#
   @method reset Continuation
   @brief Clears the continuation state.

   Allows to recycle a continuation after it is terminated, or at any moment.
   This must be called when the continuation has returned the control to its
   caller: it has no effect otherwise.

 */

FALCON_FUNC Continuation_reset ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   cc->cont()->reset();

}

/*#
   @method complete Continuation
   @brief Indicates if the continuation has been completely executed or not.
   @return True if the continuation code has exited through any mean except calling the continuation.

   When the code inside the continuation calls the continuation, it means it has more
   operations to perform; the calling code may then decide to call the continuation to let
   it continue, or to reset it, or just to discard it.
 */

FALCON_FUNC Continuation_complete ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   cc->cont()->complete();
}


FALCON_FUNC Continuation__suspend ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* susp = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   susp->cont()->suspend( vm->param(0) == 0 ? Item(): *vm->param(0) );
}

}
}

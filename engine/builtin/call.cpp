/*
   FALCON - The Falcon Programming Language.
   FILE: call.cpp

   Falcon core module -- Render function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/builtin/call.cpp"

#include <falcon/builtin/call.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/item.h>
#include <falcon/itemarray.h>

namespace Falcon {
namespace Ext {

Call::Call():
   Function( "call" )
{
   parseDescription("item:X,params:A");
}

Call::~Call()
{
}



/*#
@method call BOM
@brief Evaluates a given item passing the parameters from an array.
@optparam params Array of parameters to be sent to the function.
@return The result of the evaluation.

This function can be used to efficiently invoke a function for which
the parameters have been stored in a an array.

The called function replaces this method in the call stack, as if
it was directly called.

The following calls are equivalent:
@code
 function test(a,b)
    > "A: ", a
    > "B: ", b
 end

 test("a","b")
 [test, "a"]("b")
 test.call( ["a","b"])
 call( test, ["a","b"])
@endcode

@see passvp
*/

/*#
 @function call
 @brief Evaluates the given item passing the parameters from an array.
 @param callee Item to be evaluated.
 @optparam params Array of parameters to be sent to the evaluation.
 @return The value yielded by the evaluation.

 This function can be used to efficiently invoke a function,
 or otherwise evaluate an entity, for which
 the parameters have been stored in an array.

 The called function replaces this method in the call stack, as if
 it was directly called.

 The following calls are equivalent:
 @code
    function test(a,b)
       > "A: ", a
       > "B: ", b
    end

    test("a","b")
    [test, "a"]("b")
    call(test, ["a","b"])
 @endcode

 @see passvp
 */

void Call::invoke( VMContext* ctx, int32 )
{
   Item self;
   Item* iParams;
   if ( ctx->isMethodic() )
   {
      self = ctx->self();
      iParams = ctx->param(0);
   }
   else
   {
      self = *ctx->param(0);
      if( self == 0 )
      {
         throw paramError(__LINE__, SRC);
      }
      iParams = ctx->param(1);
   }

   if(iParams != 0 && ! iParams->isArray())
   {
      throw paramError(__LINE__, SRC);
   }

   ItemArray* ir = iParams == 0 ? 0 : iParams->asArray();
   ctx->returnFrame();
   ctx->popData();

   if( ir == 0 )
   {
      ctx->callerLine(__LINE__+1);
      ctx->callItem(self);
   }
   else {
      ItemArray local;
      // mutlitasking wise...
      local.copyOnto( *ir );
      ctx->callerLine(__LINE__+1);
      ctx->callItem( self, local.length(), local.elements() );
   }
}

}
}

/* end of call.cpp */

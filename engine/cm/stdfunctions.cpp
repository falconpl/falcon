/*
   FALCON - The Falcon Programming Language.
   FILE: stdfunctions.h

   Falcon core module -- Standard core module functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 22 Jan 2013 11:24:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/stdfunctions.cpp"

#include <falcon/trace.h>
#include <falcon/falcon.h>

#include <falcon/cm/stdfunctions.h>
#include <falcon/vm.h>
#include <falcon/vmcontext.h>
#include <falcon/sys.h>

#include <falcon/cm/iterator.h>

#include <falcon/error.h>

namespace Falcon {
namespace Ext {

/*#
   @function sleep
   @brief Put the current coroutine at sleep for some time.
   @param time Time, in seconds and fractions, that the coroutine wishes to sleep.
   @return an item posted by the embedding application.

   This function declares that the current coroutines is not willing to proceed at
   least for the given time. The VM will swap out the coroutine until the time has
   elapsed, and will make it eligible to run again after the given time lapse.

   The parameter may be a floating point number if a pause shorter than a second is
   required.
*/

FALCON_DEFINE_FUNCTION_P(sleep)
{
   TRACE1( "-- called with %d params", pCount );
   
   // all the evaluation happens in the 
   if( pCount < 1 || ! ctx->param(0)->isOrdinal() ) {
      throw paramError();
   }
   
   numeric to = ctx->param(0)->forceNumeric();
   ctx->sleep( (int64)(to * 1000) );
   ctx->returnFrame();
} 

/*#
   @function epoch
   @ingroup general_purpose
   @brief Returns the number of seconds since the "epoch" (1 Jan 1970).
   @return An integer number of seconds.
*/
FALCON_DEFINE_FUNCTION_P1(epoch)
{
   MESSAGE1( "-- called with 0 params" );
   int64 ep = Sys::_epoch();
   ctx->returnFrame(ep);
}

/*#
   @function seconds
   @ingroup general_purpose
   @brief Returns the number of seconds and milliseconds from day, activity or program start.
   @return The number of seconds and fractions of seconds in a floating point value.

   Actually, this function returns a floating point number which represents seconds
   and fraction of seconds elapse since a conventional time. This function is mainly
   meant to be used to take intervals of time inside the script,
   with a millisecond precision.
*/

FALCON_DEFINE_FUNCTION_P1(seconds)
{
   MESSAGE1( "-- called with 0 params" );
   numeric ep = Sys::_milliseconds()/1000.0;
   ctx->returnFrame(ep);
}

/*#
   @function advance
   @ingroup general_purpose
   @param collection the collection being traversed.
   @brief Advnaces an automatic iterator in the current rule context.
   @return An item from the collection.

*/

FALCON_DEFINE_FUNCTION_P(advance)
{
   TRACE1( "-- called with %d params", pCount );
   if( pCount < 1 ) {
      throw paramError();
   }

   IteratorCarrier* ic;
   Class* iterClass=0;

   if( ctx->readInit().isNil() )
   {
      iterClass = module()->getClass("Iterator");
      fassert( iterClass != 0 );
      ic = new IteratorCarrier( *ctx->param(0) );

      ctx->writeInit(FALCON_GC_STORE(iterClass, ic));
   }
   else {
      const Item& initItem = ctx->readInit();
      void* data = 0;
      initItem.asClassInst( iterClass, data );
      fassert( iterClass == module()->getClass("Iterator") );
      ic = static_cast<IteratorCarrier*>(data);
   }

   // change into the class method, and use its return frames.
   ctx->param(0)->setBoolean(false);
   static_cast<ClassIterator*>(iterClass)->invokeDirectNextMethod(ctx, ic, pCount);
}


}
}

/* end of sleep.cpp */

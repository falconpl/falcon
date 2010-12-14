/*
   FALCON - The Falcon Programming Language.
   FILE: functional_ext.cpp

   Functional programming support
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
   @funset oob_support Out of band items support
   @brief Handle out of band items.

   Out-of-band items are normal items which can be tested for the out-of-band quality
   through the @a isoob function to perform special tasks. Some core and RTL functions can
   check for the item being out-of-band to take special decisions about the item, or to
   modify their behavior. For example, the @a map function drops the item (acting like @a filter ),
   if it is out-of-band.

   This feature is available also to scripts; functions accepting any kind of items from
   callbacks they are using to generate data may wish to receive special instructions
   through out of band data. In the next example, a data producer returns a set of items
   one at a time, and notifies the caller to switch to another producer via an out-of-band
   notification.

   @code
   function firstSeries()
      static: vals = [1, 2, 3, 4 ]
      if vals: return arrayHead( vals )
      // notify the next function
      return oob( secondSeries )
   end

   function secondSeries()
      static: vals = [ "a", nil, "b", 4 ]
      if vals: return arrayHead( vals )
      // notify we're done with an nil OOB
      return oob()
   end

   function consumer( producer )
      loop item = producer()
         if isoob( item )
            // An OOB means we have something special. If it's nil, we're done...
            if item == nil: return
            // else it's the notification of a new producer
            producer = item
         else
            // if it's not an OOB, then we must process it
            > "Received item: ", item
         end
      end
   end

   consumer( firstSeries )
   @endcode

   Marking an item as out-of-band allows the creation of @i monads in functional evaluations.
   More automatism will be introduced in future, but scripters can have monads by assigning the
   oob status to complex objects and perform out-of-band processing on them.
*/

/*#
   @function oob
   @brief Generates an out-of-band item.
   @inset oob_support
   @optparam item The item to be declared out of band.
   @return An oob version of the item, or an oob @b nil if no item is given.

   This function returns an out-of-band nil item, or if a parameter is given,
   an out-of-band version of that item.
*/
FALCON_FUNC  core_oob( ::Falcon::VMachine *vm )
{
   Item *obbed = vm->param(0);
   if ( ! obbed )
   {
      vm->regA().setNil();
   }
   else {
      vm->regA() = *obbed;
   }

   vm->regA().setOob();
}


/*#
   @function deoob
   @brief Turns an out-of-band item in a normal item.
   @inset oob_support
   @param item The out of band item to be turned into a normal item.
   @return An the non-out-of-band version version of the item.

   The function returns a flat copy of the item without the out-of-band status set.
   If the item was initially not OOB, then deoob() does nothing.
   See @a oob for a deeper explanation of OOB items.
*/
FALCON_FUNC  core_deoob( ::Falcon::VMachine *vm )
{
   Item *obbed = vm->param(0);
   if ( ! obbed )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "X" ) );
      return;
   }

   vm->regA() = *obbed;
   vm->regA().resetOob();
}

/*#
   @function isoob
   @brief Checks for the out-of-band status of an item.
   @inset oob_support
   @param item The item to be checked.
   @return True if the item is out of band, false otherwise.

   This function can be used to check if a certain item is an out of band item.
*/
FALCON_FUNC  core_isoob( ::Falcon::VMachine *vm )
{
   Item *obbed = vm->param(0);
   if ( ! obbed )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "X" ) );
      return;
   }

   vm->regA().setBoolean( obbed->isOob() );
}

}
}

/* end of oob_ext.cpp */

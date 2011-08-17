/*
   FALCON - The Falcon Programming Language.
   FILE: seconds.cpp

   Time function
   -------------------------------------------------------------------
   Author: $AUTHOR
   Begin: $DATE

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/sys.h>

namespace Falcon {
namespace core {


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


FALCON_FUNC  seconds ( ::Falcon::VMachine *vm )
{
   vm->retval( Sys::_seconds() );
}

/*#
   @function epoch
   @ingroup general_purpose
   @brief Returns the number of seconds since the "epoch" (1 Jan 1970).
   @return An integer number of seconds.
*/


FALCON_FUNC  epoch ( ::Falcon::VMachine *vm )
{
   vm->retval( Sys::_epoch() );
}


}}

/* end of seconds.cpp */

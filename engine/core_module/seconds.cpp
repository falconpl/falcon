/*
   FALCON - The Falcon Programming Language.
   FILE: time.cpp

   Time function
   -------------------------------------------------------------------
   Author: $AUTHOR
   Begin: $DATE

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/vm.h>

#ifndef WIN32
#include <sys/time.h>
#else
#include <windows.h>
#endif

namespace Falcon {
namespace core {


/*#
   @function seconds
   @ingroup general_purpose
   @brief Returns the number of seconds since the "epoch" as reported by the system.
   @return The number of seconds and fractions of seconds in a floating point value.

   Actually, this function returns a floating point number which represents seconds
   and fraction of seconds elapse since a conventional time. This function is mainly
   meant to be used to take intervals of time inside the script,
   with a millisecond precision.
*/


FALCON_FUNC  seconds ( ::Falcon::VMachine *vm )
{
#ifdef WIN32
   SYSTEMTIME SysTime ;
   GetSystemTime( &SysTime );
   vm->retval( SysTime.wHour * 3600.0 + SysTime.wMinute *60 +
         SysTime.wSecond + SysTime.wMilliseconds /1000.0 );

#else
   struct timeval time;
   gettimeofday( &time, 0 );
   numeric secs = time.tv_sec + (time.tv_usec / 1000000.0 );
   vm->retval( secs );
#endif
}

}}

/* end of time.cpp */

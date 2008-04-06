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

namespace Falcon { namespace Ext {


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

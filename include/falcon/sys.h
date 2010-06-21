/*
   FALCON - The Falcon Programming Language.
   FILE: flc_sys.h

   System related services.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar nov 9 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System related services.
*/

#ifndef flc_flc_sys_H
#define flc_flc_sys_H

#include <falcon/types.h>
#include <falcon/dir_sys.h>
#include <falcon/time_sys.h>

namespace Falcon {

class String;

namespace Sys {


/** Gives current second count from Epoch.
   The number of seconds is generally returned in GMT, if this
   feature is available in the system.
   \return a float nubmer, where decimals are up to milliseconds.
*/
FALCON_DYN_SYM numeric _seconds();

/** Gives current second count from Epoch in localtime.
   The number of seconds is generally returned in localtime, if this
   feature is available in the system.
   \return a float nubmer, where decimals are up to milliseconds.
*/
FALCON_DYN_SYM numeric _localSeconds();

/** Returns a counter representing a time entity.
   This is a direct interface to the fastest possible
   function the hosts systems has to provide a millisecond
   counter. The "0" moment is undefined (it may be the start
   of the program, the last reboot time or epoch), and the
   resolution of the calls are limited to what the system
   provides, but it is granted that calling _milliseconds()
   at T1 > T0 will result in the return value of the second
   call being greater or equal than the first. The difference
   between the two numbers express the number of milliseconds
   roughly elapsed between the two calls.

   \return millisecond counter value.
*/
FALCON_DYN_SYM uint32 _milliseconds();

/** Returns a valid and possibly unique temporary file name.
   Just a haky test for now, final version must OPEN the stream and return it.
   \param res on return will contain a to C stringed filename.
*/
FALCON_DYN_SYM void _tempName( ::Falcon::String &res );

FALCON_DYN_SYM bool _describeError( int64 eid, String &target );
FALCON_DYN_SYM int64 _lastError();
FALCON_DYN_SYM bool _getEnv( const String &var, String &result );
FALCON_DYN_SYM bool _setEnv( const String &var, const String &value );
FALCON_DYN_SYM bool _unsetEnv( const String &var );

/** Returns seconds since epoch.
   Used in many systems.
*/

FALCON_DYN_SYM int64 _epoch();

/** Callback for environment variable enumeration.
 \see _enumerateEnvironment
*/
typedef void(*EnvStringCallback)( const String& key, const String& value, void* cbData );

/** Gets all the environemnt string of the calling process.

   Each pair of key/value pair is sent to the cb function for
   processing. If possible, the strings are already rendered from
   local encoding.

   The cbData parameter is an arbitrary data that is passed to the CB function
   for processing.

*/
FALCON_DYN_SYM void _enumerateEnvironment( EnvStringCallback cb, void* cbData );

FALCON_DYN_SYM void _dummy_ctrl_c_handler();

#ifdef FALCON_SYSTEM_WIN
}
}
   #include <windows.h>
   #include <falcon/string.h>
namespace Falcon {
namespace Sys {
   FALCON_DYN_SYM numeric SYSTEMTIME_TO_SECONDS( const SYSTEMTIME &st );
#endif

}
}

#endif

/* end of flc_sys.h */

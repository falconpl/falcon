/*
   The Falcon Programming Language
   FILE: dynlib_sys_dl.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - UNIX dload() specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   System specific extensions.
*/

#include "dynlib_sys.h"

#include <falcon/autocstring.h>
#include <dlfcn.h>
#include <errno.h>

namespace Falcon {
namespace Sys {

void *dynlib_load( const String &libpath )
{
   AutoCString c_path( libpath );
   return dlopen( c_path.c_str(), RTLD_LAZY );
}

int dynlib_unload( void *libhandler )
{
   return dlclose(libhandler);
}

void *dynlib_get_address( void *libhandler, const String &func_name )
{
   AutoCString c_sym( func_name );
   return dlsym( libhandler, c_sym.c_str() );
}

bool dynlib_get_error( int32 &ecode, String &sError )
{
   const char *error = dlerror();
   // no error? -- don't mangle the string and return false.
   if ( error == 0 )
      return false;

   // bufferize the error and signal we have some.
   ecode = errno;
   sError.bufferize( error );
   return true;
}


}
}

/* end of dynlib_sys_dl.cpp */

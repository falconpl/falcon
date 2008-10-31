/*
   The Falcon Programming Language
   FILE: dynlib_sys.h

   Direct dynamic library interface for Falcon
   System specific extensions
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

#ifndef dynlib_sys_H
#define dynlib_sys_H

#include <falcon/setup.h>
#include <falcon/string.h>

namespace Falcon {
namespace Sys {
   void *dynlib_load( const String &libpath );
   int dynlib_unload( void *libhandler );
   void *dynlib_get_address( void *libhandler, const String &func_name );
   bool dynlib_get_error( int32 &ecode, String &sError );
}
}

#endif

/* end of dynlib_sys.h */

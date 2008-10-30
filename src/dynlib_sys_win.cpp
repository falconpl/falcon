/*
   The Falcon Programming Language
   FILE: dynlib_sys_win.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - MS-Windows specific extensions
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

namespace Falcon {
namespace Sys {

void *dynlib_load( const String &libpath )
{
}

int dynlib_unload( void *libhandler )
{
   return 0;
}

void *dynlib_get_address( void *libhandler, const String &func_name )
{
}

bool dynlib_get_error( String &error )
{
   return false;
}

}
}

/* end of dynlib_sys_win.cpp */

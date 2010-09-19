/*
   The Falcon Programming Language
   FILE: dynlib_sys_posix.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - UNIX dload() + POSIX so.
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

const char* dynlib_get_dynlib_ext()
{
   static const char* ext = "so";
   return ext;
}

}
}

/* end of dynlib_sys_posix.cpp */

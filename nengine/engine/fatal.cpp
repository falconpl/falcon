/*
   FALCON - The Falcon Programming Language.
   FILE: fatal.cpp

   Function signaling a fatal error (which should abort).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Mar 2011 16:44:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/fatal.h>

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

namespace Falcon {

static void _fatal( const char* msg, ... )
{
  va_list args;
  va_start( args, msg);
  vfprintf( stderr, msg, args );
  va_end (args);  
  
  abort();
}

void(*fatal)( const char* msg, ... ) = _fatal;

}

/* end of fatal.cpp */

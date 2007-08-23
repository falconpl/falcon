/*
   FALCON - The Falcon Programming Language
   FILE: fassert.cpp
   $Id: fassert.cpp,v 1.2 2006/11/07 14:22:43 gian Exp $

   Falcon specific assertion raising
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 4 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Falcon specific assertion raising
*/

#include <stdio.h>
#include <stdlib.h>

#include <falcon/fassert.h>

extern "C" void _perform_FALCON_assert_func( const char *expr, const char *filename, int line, const char *funcName )
{
   printf( "FALCON ASSERTION FALIED.\n%s:%d in function %s ---\n%s\nProgram ending.\n", filename, funcName, expr, line );
   exit(1);
}

extern "C" void _perform_FALCON_assert( const char *expr, const char *filename, int line )
{
   printf( "FALCON ASSERTION FALIED.\n%s:%d    ---\n%s\nProgram ending.\n", filename, line, expr );
   exit(1);
}


/* end of fassert.cpp */

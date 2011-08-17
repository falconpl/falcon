/*
   FALCON - The Falcon Programming Language
   FILE: fassert.cpp

   Falcon specific assertion raising
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 4 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon specific assertion raising
*/

#include <stdio.h>
#include <stdlib.h>

#include <falcon/fassert.h>

extern "C" void _perform_FALCON_assert_func( const char *expr, const char *filename, int line, const char *funcName )
{
   printf( "FALCON ASSERTION FALIED.\n%s:%d in function %s ---\n%s\nProgram ending.\n", filename, line, funcName, expr );
   abort();
}

extern "C" void _perform_FALCON_assert( const char *expr, const char *filename, int line )
{
   printf( "FALCON ASSERTION FALIED.\n%s:%d    ---\n%s\nProgram ending.\n", filename, line, expr );
   abort();
}


/* end of fassert.cpp */

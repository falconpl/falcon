/*
   FALCON - The Falcon Programming Language.
   FILE: flc_memory.cpp

   Basic memory manager functions and function pointers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/memory.h>
#if defined(__BORLANDC__)
   #include <stdio.h>
   #include <stdlib.h>
#else
   #include <cstdio>
   #include <cstdlib>
#endif

namespace Falcon {
/*
#ifdef _MSC_VER
const builder_t Builder;
#else
const builder_t Builder = {};
#endif
*/
void * DflMemAlloc( size_t amount )
{
   void *ret = malloc( amount );
   if ( ret == 0 ) {
      printf( "Falcon: fatal allocation error when allocating %d bytes\n", amount );
      exit(1);
   }
   return ret;
}

void DflMemFree( void *mem )
{
   free( mem );
}

void * DflMemRealloc( void *mem, size_t amount )
{
   void *ret = realloc( mem, amount );


   if ( ret == 0 && amount != 0 ) {
      printf( "Falcon: fatal reallocation error when allocating %d bytes\n", amount );
      exit(1);
   }
   return ret;
}

void * (*memAlloc) ( size_t ) = DflMemAlloc;
void (*memFree) ( void * ) = DflMemFree;
void * (*memRealloc) ( void *,  size_t ) = DflMemRealloc;
}

/* end of flc_memory.cpp */

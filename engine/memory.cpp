/*
   FALCON - The Falcon Programming Language.
   FILE: flc_memory.cpp
   $Id: memory.cpp,v 1.3 2007/01/24 21:49:55 jonnymind Exp $

   Basic memory manager functions and function pointers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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


   if ( ret == 0 ) {
      printf( "%s\n", "Falcon: fatal allocation error" );
      exit(1);
   }
   return ret;
}

void * (*memAlloc) ( size_t ) = DflMemAlloc;
void (*memFree) ( void * ) = DflMemFree;
void * (*memRealloc) ( void *,  size_t ) = DflMemRealloc;
}

/* end of flc_memory.cpp */

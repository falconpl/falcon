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
#if defined(__SUNPRO_CC)
   #include <stdio.h>
#elif defined(__BORLANDC__)
   #include <stdio.h>
   #include <stdlib.h>
#else
   #include <cstdio>
   #include <cstdlib>
#endif

#ifdef FALCON_SYSTEM_WIN
#include <falcon/mt_win.h>
#else
#include <falcon/mt_posix.h>
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
   void *ret = malloc( amount + sizeof(size_t) );
   if ( ret == 0 ) {
      printf( "Falcon: fatal allocation error when allocating %d bytes\n", (int) amount );
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
      printf( "Falcon: fatal reallocation error when allocating %d bytes\n", (int) amount );
      exit(1);
   }

   return ret;
}


void * DflAccountMemAlloc( size_t amount )
{
   size_t *ret =  (size_t*) malloc( amount + sizeof(size_t) );
   if ( ret == 0 ) {
      printf( "Falcon: fatal allocation error when allocating %d bytes\n", (int) amount );
      exit(1);
   }

   gcMemAccount( amount );
   *ret = amount;
   return ret+1;
}


void DflAccountMemFree( void *mem )
{
   if ( mem != 0 )
   {
      size_t *smem = (size_t*) mem;
      gcMemUnaccount( smem[-1] );
      free( smem-1 );
   }
}

void * DflAccountMemRealloc( void *mem, size_t amount )
{
   if ( amount == 0 )
   {
      DflAccountMemFree( mem );
      return 0;
   }

   if ( mem == 0 )
      return DflAccountMemAlloc( amount );


   size_t *smem = (size_t*) mem;
   smem--;
   size_t oldalloc = *smem;

   size_t *nsmem = (size_t*) realloc( smem, amount + sizeof( size_t ) );

   if ( nsmem == 0 ) {
      printf( "Falcon: fatal reallocation error when allocating %d bytes\n", (int) amount );
      exit(1);
   }

   *nsmem = amount;
   if( amount > oldalloc )
      gcMemAccount( amount - oldalloc );
   else
      gcMemUnaccount( oldalloc - amount );

   return nsmem+1;
}


//===================================================================================
// Account functions
//
static Mutex *s_gcMutex = 0;
static size_t s_allocatedMem  = 0;

void gcMemAccount( size_t mem )
{
   if( s_gcMutex == 0 )
      s_gcMutex = new Mutex;

   s_gcMutex->lock();
   s_allocatedMem += mem;
   s_gcMutex->unlock();
}

void gcMemUnaccount( size_t mem )
{
   if( s_gcMutex == 0 )
      s_gcMutex = new Mutex;

   s_gcMutex->lock();
   s_allocatedMem -= mem;
   s_gcMutex->unlock();
}

size_t gcMemAllocated()
{
   if( s_gcMutex == 0 )
      s_gcMutex = new Mutex;

   s_gcMutex->lock();
   register uint32 val = s_allocatedMem;
   s_gcMutex->unlock();

   return val;
}

void gcMemShutdown()
{
	delete s_gcMutex;
	s_gcMutex = 0;
}

//============================================================
// Global function pointers.
//
void * (*gcAlloc) ( size_t ) = DflAccountMemAlloc;
void (*gcFree) ( void * ) = DflAccountMemFree;
void * (*gcRealloc) ( void *,  size_t ) = DflAccountMemRealloc;

/*
void * (*memAlloc) ( size_t ) = DflMemAlloc;
void (*memFree) ( void * ) = DflMemFree;
void * (*memRealloc) ( void *,  size_t ) = DflMemRealloc;
*/
/*
   In phase 1, we're accounting everything to verify the basic solidity of our GC system.
*/
void * (*memAlloc) ( size_t ) = DflAccountMemAlloc;
void (*memFree) ( void * ) = DflAccountMemFree;
void * (*memRealloc) ( void *,  size_t ) = DflAccountMemRealloc;

}

/* end of flc_memory.cpp */

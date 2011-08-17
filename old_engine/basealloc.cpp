/*
   FALCON - The Falcon Programming Language
   FILE: basealloc.cpp

   Base allocation declaration for engine classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar dic 5 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
    Base allocation declaration for engine classes
*/

#include <falcon/setup.h>
#include <falcon/globals.h>
#include <falcon/memory.h>
#include <falcon/basealloc.h>

namespace Falcon {

void *BaseAlloc::operator new( size_t size )
{
   return memAlloc( size );
}

void BaseAlloc::operator delete( void *mem, size_t /* currenty unused */ )
{
   memFree( mem );
}

}


/* end of basealloc.cpp */

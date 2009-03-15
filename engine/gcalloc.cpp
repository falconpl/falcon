/*
   FALCON - The Falcon Programming Language
   FILE: gcalloc.cpp

   Base allocation declaration for engine classes (gc sensible)
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
#include <falcon/memory.h>
#include <falcon/gcalloc.h>
#include <falcon/globals.h>

namespace Falcon {

void *GCAlloc::operator new( size_t size )
{
   return gcAlloc( size );
}

void GCAlloc::operator delete( void *mem, size_t /* currenty unused */ )
{
   gcFree( mem );
}

}


/* end of gcalloc.cpp */

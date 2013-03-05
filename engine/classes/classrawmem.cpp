/*
   FALCON - The Falcon Programming Language.
   FILE: classrawmem.h

   Handler for unformatted raw memory stored in the GC.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Jun 2012 16:52:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classrawmem.cpp"

#include <falcon/classes/classrawmem.h>
namespace Falcon {


ClassRawMem::ClassRawMem():
   Class("RawMem")
{
}

ClassRawMem::~ClassRawMem()
{
}

void* ClassRawMem::createInstance() const
{
   return 0;
}

void ClassRawMem::dispose( void* instance ) const
{
   delete[] (byte*) instance;
}

void* ClassRawMem::clone( void* ) const
{
   return 0;
}


void ClassRawMem::gcMarkInstance( void* instance, uint32 mark ) const
{
   *((uint32*) instance) = mark;
}

bool ClassRawMem::gcCheckInstance( void* instance, uint32 mark ) const
{
   return *((uint32*)instance) >= mark;
}

void* ClassRawMem::allocate( uint32 size ) const
{
   return new byte[size];
}

}

/* end of classrawmem.cpp */


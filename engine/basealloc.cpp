/*
   FALCON - The Falcon Programming Language
   FILE: basealloc.cpp
   $Id: basealloc.cpp,v 1.1 2006/12/05 15:28:47 gian Exp $

   Base allocation declaration for engine classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar dic 5 2006
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
    Base allocation declaration for engine classes
*/

#include <falcon/setup.h>
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

/*
   FALCON - The Falcon Programming Language.
   FILE: gcalloc.h

   Base allocation declaration for engine classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Mar 2009 14:17:48 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base allocation declaration for engine classes
*/

#ifndef flc_gcalloc_H
#define flc_gcalloc_H

#include <falcon/setup.h>
#include <stdlib.h>  // for size_t declaration

namespace Falcon {

class FALCON_DYN_CLASS GCAlloc
{
public:
   void *operator new( size_t size );
   void operator delete( void *mem, size_t size );
};

}

#endif

/* end of gcalloc.h */

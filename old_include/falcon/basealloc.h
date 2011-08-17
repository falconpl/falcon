/*
   FALCON - The Falcon Programming Language.
   FILE: basealloc.h

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

#ifndef flc_basealloc_H
#define flc_basealloc_H

#include <falcon/setup.h>
#include <stdlib.h>  // for size_t declaration

namespace Falcon {

class FALCON_DYN_CLASS BaseAlloc
{
public:
   void *operator new( size_t size );
   void operator delete( void *mem, size_t size );
};

}

#endif

/* end of basealloc.h */

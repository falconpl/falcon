/*
   FALCON - The Falcon Programming Language.
   FILE: basealloc.h
   $Id: basealloc.h,v 1.1 2006/12/05 15:28:46 gian Exp $

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

/*
   FALCON - The Falcon Programming Language.
   FILE: destroyable.h

   Base class for user-defined pointers in objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab feb 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Base class for user-defined pointers in objects.
*/

#ifndef flc_destroyable_H
#define flc_destroyable_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/basealloc.h>

namespace Falcon {

/** Vitual destructor stub class for user defined pointers.
   The Destroyable class is a simple stub class that provides a virtual
   destroyer and a signature field. The user willing to set private data
   into objecs (CoreObject class) will have to instantiate one item
   from this class and set it into the object. When the object is destroyed,
   the destroyable virtual destructor is called with him.
*/

class FALCON_DYN_CLASS Destroyable: public BaseAlloc
{
public:
   virtual ~Destroyable();
};

}

#endif

/* end of destroyable.h */

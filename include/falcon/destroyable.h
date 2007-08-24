/*
   FALCON - The Falcon Programming Language.
   FILE: destroyable.h
   $Id: destroyable.h,v 1.2 2006/12/05 15:28:46 gian Exp $

   Base class for user-defined pointers in objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab feb 25 2006
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
   virtual ~Destroyable() {}
};

}

#endif

/* end of destroyable.h */

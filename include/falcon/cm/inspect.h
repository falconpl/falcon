/*
   FALCON - The Falcon Programming Language.
   FILE: inspect.h

   Falcon core module -- deep recursive exploration of objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 11:12:52 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_INSPECT_H
#define FALCON_CORE_INSPECT_H

#include <falcon/function.h>
#include <falcon/pstep.h>
#include <falcon/string.h>

#include <falcon/fassert.h>

#include <falcon/trace.h>

namespace Falcon {
namespace Ext {

/*#
   @function inspect
   @inset core_basic_io
   @param item An item to be inspected
   @optparam maxdepth Maximum depth of the printout -- defaults to 3
   @optparam maxsize maximum size of the single printout 
   @brief Verbosely print the contents of an item to the standard error stream.

   This function deeply inspects an item by recursively inspecting each public
   property, including functions, methods and classes.
 
   @see describe
*/

class FALCON_DYN_CLASS Inspect: public Function
{
public:   
   Inspect();
   virtual ~Inspect();
   virtual void invoke( VMContext* ctx, int32 nParams );
};

}
}

#endif	

/* end of inspect.h */

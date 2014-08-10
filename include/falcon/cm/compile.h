/*
   FALCON - The Falcon Programming Language.
   FILE: compile.h

   Falcon core module -- Dynamic compile function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 13 Jan 2012 15:12:11 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_COMPILE_H
#define FALCON_CORE_COMPILE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {


class FALCON_DYN_CLASS Compile: public Function
{
public:   
   Compile();
   virtual ~Compile();
   virtual void invoke( VMContext* ctx, int32 nParams );
};

}
}

#endif	

/* end of compile.h */

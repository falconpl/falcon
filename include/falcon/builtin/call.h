/*
   FALCON - The Falcon Programming Language.
   FILE: call.h

   Falcon core module -- call function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_CALL_H
#define FALCON_CORE_CALL_H

#include <falcon/function.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS Call: public Function
{
public:
   Call();
   virtual ~Call();
   virtual void invoke( VMContext* vm, int32 nParams );
};

}
}

#endif

/* end of call.h */

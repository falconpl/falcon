/*
   FALCON - The Falcon Programming Language.
   FILE: vmsema.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio nov 11 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#ifndef flc_vmsema_H
#define flc_vmsema_H

#include <falcon/vm.h>
#include <falcon/userdata.h>

namespace Falcon {

class VMContext;

class VMSemaphore: public UserData
{
   int32 m_count;
   ContextList m_waiting;

public:
   VMSemaphore( int32 count = 0 ):
      m_count( count )
   {}

   ~VMSemaphore() {}

   void post( VMachine *vm, int32 value=1 );
   void wait( VMachine *vm, double time = -1.0 );

   void unsubscribe( VMContext *ctx );
};

}

#endif

/* end of vmsema.h */

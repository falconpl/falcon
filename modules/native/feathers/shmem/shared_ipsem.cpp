/*
   FALCON - The Falcon Programming Language.
   FILE: shared_ipsem.cpp
   Interface for the Falcon VM to a shared IP semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/shmem/shared_ipsem.cpp"

#include "shared_ipsem.h"
#include "ipsem_ext.h"

namespace Falcon {

SharedIPSem::SharedIPSem( ContextManager* ctx, Class* handler ):
      Shared(ctx, handler)
{
}

SharedIPSem::SharedIPSem( ContextManager* ctx, Class* handler, const String& name ):
      Shared(ctx, handler),
      m_sem(name)
{
}

SharedIPSem::SharedIPSem( const SharedIPSem& other ):
         Shared( other.notifyTo(), other.handler() ),
         m_sem(other.m_sem)
{

}


SharedIPSem::~SharedIPSem()
{
}

int32 SharedIPSem::consumeSignal( VMContext* target, int32 count )
{
   int i = 0;
   for( ; i < count; ++i )
   {
      if ( ! m_sem.tryWait() )
      {
         return Shared::consumeSignal(target, count - i) + i;
      }
   }

   return i;
}


void SharedIPSem::onWaiterWaiting(VMContext*, int64 to)
{
   // we ask our class to wait for the IPC semaphore to be ready.
   const ClassIPSem* cs = static_cast<const ClassIPSem*>(handler());
   cs->waitOn(this, to);
}


}


/* end of shared_ipsem.cpp */


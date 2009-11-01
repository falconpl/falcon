/*
   FALCON - The Falcon Programming Language.
   FILE: vmsema.cpp

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
#include "vmsema.h"
#include <falcon/vmcontext.h>

namespace Falcon {

void VMSemaphore::post( VMachine *vm, int32 count )
{
   m_count += count;

   while( m_count > 0 && ! m_waiting.empty() )
   {
      VMContext *ctx = (VMContext *) m_waiting.front();
      ctx->signaled();
      vm->reschedule( ctx );

      m_waiting.popFront();
      m_count --;
   }
}

void VMSemaphore::wait( VMachine *vm, numeric to )
{
   if ( m_count == 0 ) {
      m_waiting.pushBack( vm->m_currentContext );
      vm->m_currentContext->waitOn( this, to );
      // by default will be zero; 1 if correctly awaken
      vm->regA().setBoolean( false );
      // now we can change the context
      vm->rotateContext();
   }
   else {
      vm->regA().setBoolean( true );
      m_count --;
   }
}

void VMSemaphore::unsubscribe( VMContext *ctx )
{
   ListElement *elem = m_waiting.begin();
   while( elem != 0 )
   {
      VMContext *cty = (VMContext *) elem->data();
      if ( ctx == cty )
      {
         m_waiting.erase( elem );
         return;
      }
      elem = elem->next();
   }
}

VMSemaphore *VMSemaphore::clone() const
{
   // ! not supported.
   return 0;
}

}


/* end of vmsema.cpp */

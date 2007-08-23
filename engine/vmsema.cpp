/*
   FALCON - The Falcon Programming Language.
   FILE: vmsema.cpp
   $Id: vmsema.cpp,v 1.3 2007/04/23 14:57:35 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio nov 11 2004
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
   Short description
*/
#include "vmsema.h"
#include <falcon/vmcontext.h>

namespace Falcon {

void VMSemaphore::post( VMachine *vm, int32 count )
{
   m_count += count;
   while( m_count > 0 && ! m_waiting.empty() ) {
      vm->putAtSleep( (VMContext *) m_waiting.front(), 0.0 );
      m_waiting.popFront();
      m_count --;
   }
}

void VMSemaphore::wait( VMachine *vm )
{
   if ( m_count == 0 ) {
      m_waiting.pushBack( vm->m_currentContext );
      vm->m_event = VMachine::eventWait;
   }
   else
      m_count --;


}

}


/* end of vmsema.cpp */

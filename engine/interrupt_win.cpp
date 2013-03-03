/*
   FALCON - The Falcon Programming Language.
   FILE: interrupt_win.cpp

   Implements VM interruption protocol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 12:25:12 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/fatal.h>
#include <falcon/interrupt.h>
#include <windows.h>


namespace Falcon {

Interrupt::Interrupt():
   m_sysdata( 0 )
{
   m_sysdata = ::CreateEvent( NULL, TRUE, FALSE, NULL );
}


Interrupt::~Interrupt()
{
   ::CloseHandle( (HANDLE) m_sysdata );
}


void Interrupt::interrupt()
{
   HANDLE hEvent = (HANDLE) m_sysdata;
   ::SetEvent( hEvent );
}

bool Interrupt::interrupted() const
{
   HANDLE hEvent = (HANDLE) m_sysdata;
   return ::WaitForSingleObject(hEvent, 0) == WAIT_OBJECT_0;
}

void Interrupt::reset()
{
   HANDLE hEvent = (HANDLE) m_sysdata;
   ::ResetEvent( hEvent );
}

}

/* end of interrupt_cpp.h */

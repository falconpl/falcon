/*
   FALCON - The Falcon Programming Language.
   FILE: vm_sys_win.cpp

   System specifics for the falcon VM - POSIX compliant systems.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Apr 2008 17:30:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/vm_sys.h>
#include <falcon/vm_sys_win.h>
#include <falcon/memory.h>

namespace Falcon {
namespace Sys {

SystemData::SystemData()
{
   m_sysData = (struct VM_SYS_DATA*) memAlloc( sizeof( struct VM_SYS_DATA ) );

   // create our interrupting event (a manual reset event)
   m_sysData->evtInterrupt = CreateEvent( NULL, TRUE, FALSE, NULL );
}


SystemData::~SystemData()
{
   CloseHandle( m_sysData->evtInterrupt );
   // delete the structure
   memFree( m_sysData );
}


bool SystemData::interrupted() const
{
   return WaitForSingleObject( m_sysData->evtInterrupt, 0 ) == WAIT_OBJECT_0;
}


void SystemData::interrupt()
{
   SetEvent( m_sysData->evtInterrupt );
}

void SystemData::resetInterrupt()
{
   ResetEvent( m_sysData->evtInterrupt );
}


bool SystemData::sleep( numeric seconds ) const
{
   return WaitForSingleObject( m_sysData->evtInterrupt, (DWORD) (seconds * 1000.0) ) != WAIT_OBJECT_0;
}


const char *SystemData::getSystemType()
{
   return "WIN";
}

}
}

/* end of vm_sys_win.cpp */

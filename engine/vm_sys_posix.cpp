/*
   FALCON - The Falcon Programming Language.
   FILE: vm_sys_posix.cpp

   System specifics for the falcon VM - POSIX compliant systems.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Apr 2008 17:30:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/vm_sys.h>
#include <falcon/vm_sys_posix.h>
#include <falcon/memory.h>
#include <unistd.h>
#include <poll.h>
#include <stropts.h>
#include <errno.h>

namespace Falcon {
namespace Sys {

SystemData::SystemData()
{
   m_sysData = (struct VM_SYS_DATA*) memAlloc( sizeof( struct VM_SYS_DATA ) );

   // create the dup'd suspend pipe.
   pipe( m_sysData->interruptPipe );
}


SystemData::~SystemData()
{
   // close the pipe
   close( m_sysData->interruptPipe[0] );
   close( m_sysData->interruptPipe[1] );

   // delete the structure
   memFree( m_sysData );
}


bool SystemData::interrupted()
{
   int res;
   struct pollfd fds[1];
   fds[0].events = POLLIN;
   fds[0].fd = m_sysData->interruptPipe[0];

   while( (res = poll( fds, 1, 0 ) ) == EAGAIN );
   // If we have one event, the we have to read...
   return res == 1;
}


void SystemData::interrupt()
{
   write( m_sysData->interruptPipe[1], "0", 1 );
}

void SystemData::resetInterrupt()
{
   int res;
   struct pollfd fds[1];
   fds[0].events = POLLIN;
   fds[0].fd = m_sysData->interruptPipe[0];

   do {
      res = poll( fds, 1, 0 );
      if( res == 1 )
      {
         char dt;
         read( m_sysData->interruptPipe[0], &dt, 1 );
         continue;
      }

   } while( res == EAGAIN );

}


bool SystemData::sleep( numeric seconds )
{
   int ms = (int) seconds * 1000;
   struct pollfd fds[1];
   fds[0].events = POLLIN;
   fds[0].fd = m_sysData->interruptPipe[0];

   int res;
   while( ( res = poll( fds, 1, ms )) == EAGAIN );

   // if res is 0, then we completed the wait.
   return res == 0;
}


const char *SystemData::getSystemType()
{
   return "POSIX";
}

}
}

/* end of vm_sys_posix.cpp */

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
#include <falcon/signals.h>
#include <unistd.h>
#include <poll.h>
#include <stdio.h>

// if poll.h does not define POLLIN, then it's in stropts.
#ifndef POLLIN
   #include <stropts.h>
#endif
#include <errno.h>

namespace Falcon {
namespace Sys {

SystemData::SystemData(VMachine *vm)
{
   m_sysData = (struct VM_SYS_DATA*) memAlloc( sizeof( struct VM_SYS_DATA ) );

   m_vm = vm;
   m_sysData->isSignalTarget = false;

   // create the dup'd suspend pipe.
   if( pipe( m_sysData->interruptPipe ) != 0 )
   {
      printf( "Falcon: fatal allocation error in creating pipe at %s:%d\n", __FILE__, __LINE__ );
      exit(1);
   }
}


SystemData::~SystemData()
{
   // close the pipe
   close( m_sysData->interruptPipe[0] );
   close( m_sysData->interruptPipe[1] );

   // delete the structure
   memFree( m_sysData );
}


bool SystemData::interrupted() const
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
   if( write( m_sysData->interruptPipe[1], "0", 1 ) != 1 )
   {
      printf( "Falcon: fatal error in writing to the interrupt pipe at %s:%d\n", __FILE__, __LINE__ );
      exit(1);
   }

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
         if( read( m_sysData->interruptPipe[0], &dt, 1 ) < 0 )
         {
            printf( "Falcon: fatal error in reading from the interrupt pipe at %s:%d\n", __FILE__, __LINE__ );
            exit(1);
         }
         continue;
      }

   } while( res == EAGAIN );

}


bool SystemData::sleep( numeric seconds ) const
{
   int ms = (int) (seconds * 1000.0);
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


bool SystemData::becomeSignalTarget()
{
   if ( 0 != signalReceiver )
      return false;

   m_sysData->isSignalTarget = true;
   signalReceiver = new SignalReceiver(m_vm);
   signalReceiver->start();

   return true;
}


void SystemData::earlyCleanup()
{
   if ( m_sysData->isSignalTarget ) {
      delete signalReceiver;
      signalReceiver = 0;
   }
}

}
}

/* end of vm_sys_posix.cpp */

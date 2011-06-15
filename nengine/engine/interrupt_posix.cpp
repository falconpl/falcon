/*
   FALCON - The Falcon Programming Language.
   FILE: interrupt_posix.cpp

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
#include <unistd.h>
#include <poll.h>
#include <stdio.h>

// if poll.h does not define POLLIN, then it's in stropts.
#ifndef POLLIN
   #include <stropts.h>
#endif
#include <errno.h>

namespace Falcon {

Interrupt::Interrupt():
   m_sysdata( 0 )
{
   int* fdes = new int[2];
   m_sysdata = fdes;

   // create the dup'd suspend pipe.
   if( ::pipe( fdes ) != 0 )
   {
      MESSAGE( "Failed to create a pipe -- terminating" );
      fatal( "Falcon: fatal allocation error in creating pipe at %s:%d\n", __FILE__, __LINE__ );
   }
}


Interrupt::~Interrupt()
{
   int* fdes = (int*)m_sysdata;
   // close the pipe
   ::close( fdes[0] );
   ::close( fdes[1] );

   delete[] fdes;
}


void Interrupt::interrupt()
{
   int* fdes = (int*)m_sysdata;

   if( ::write( fdes[1], "0", 1 ) != 1 )
   {
      MESSAGE( "Failed to create a pipe -- terminating" );
      fatal( "Falcon: fatal write error in writing interrupt pipe at %s:%d\n", __FILE__, __LINE__ );
   }
}

bool Interrupt::interrupted() const
{
   int res;
   int* fdes = (int*)m_sysdata;
   struct pollfd fds[1];
   fds[0].events = POLLIN;
   fds[0].fd = fdes[0];

   while( (res = poll( fds, 1, 0 ) ) == EAGAIN );
   // If we have one event, the we have to read...
   return res == 1;
}

void Interrupt::reset()
{
   int res;
   struct pollfd fds[1];
   int* fdes = (int*)m_sysdata;

   fds[0].events = POLLIN;
   fds[0].fd = fdes[0];

   do {
      res = ::poll( fds, 1, 0 );
      if( res == 1 )
      {
         char dt;
         if( ::read( fdes[0], &dt, 1 ) < 0 )
         {
            MESSAGE( "Failed to read the interrupt pipe" );
            fatal( "Falcon: fatal error in reading from the interrupt pipe at %s:%d\n", __FILE__, __LINE__ );
         }
         continue;
      }

   } while( res == EAGAIN );
}

}

/* end of interrupt_posix.cpp */

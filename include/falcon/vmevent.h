/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: VMEvent.h

   Special exception to communicate relevant events.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 08 Jul 2009 14:18:55 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_VMEVENT_H
#define FALCON_VMEVENT_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>

namespace Falcon
{

/** Virtual machine events
   This class is thrown inside the VM to communicate special
   requests to the listening application.
   
   Type of requests are:
   - evQuit: Clean exit. The VM terminated.
   - evReturn: Some inner function asked immediate suspension of the VM;
               the VM is in a coherent state and can be reentered in any moment.
   - evOpLimit: The virtual machine exited due to excessive iterations.
   - evMemLimit: The virtual machine exited due to excessive memory consumption.
   - evDepthLimit: The virtual machine exited due to excessive depth of the stack calls.
*/

class VMEvent: public BaseAlloc
{

public:
   typedef enum {
      evQuit,
      evOpLimit,
      evMemLimit,
      evDepthLimit
   } tEvent;

   VMEvent( tEvent t ):
      m_type( t )
   {}
   
   VMEvent( const VMEvent& other ):
      m_type( other.m_type )
   {}
   
   ~VMEvent() {}
   
   tEvent type() const { return m_type; }
   
private:
   tEvent m_type;
};


class VMEventQuit: public VMEvent
{
public:
   VMEventQuit(): VMEvent( evQuit )
   {}
};

}

#endif

/* end of vmevent.h */

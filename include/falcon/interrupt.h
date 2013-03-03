/*
   FALCON - The Falcon Programming Language.
   FILE: interrupt.h

   Implements VM interruption protocol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 12:25:12 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_INTERRUPT_H
#define	_FALCON_INTERRUPT_H

namespace Falcon {

/** VM Interruption protocol.

 This class is used by the VM and by all the blocking or waiting operations
 that are or may be managed by a VM to interrupt waits.

 */
class FALCON_DYN_CLASS Interrupt
{
public:
   Interrupt();
   ~Interrupt();

   void interrupt();
   void reset();
   bool interrupted() const;

   /** Returns the system specific data of this interruptor.

    Used by low-level functions.
    */
   void *sysData() const { return m_sysdata; }

private:
   void* m_sysdata;
};

}

#endif	/* _FALCON_INTERRUPT_H */

/* end of interrupt.h */

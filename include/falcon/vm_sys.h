/*
   FALCON - The Falcon Programming Language.
   FILE: vm_sys.h

   Implementation of virtual - non main loop
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Apr 2008 17:30:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FLC_VM_SYS_H
#define FLC_VM_SYS_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {
namespace Sys {

struct VM_SYS_DATA;

/** System specific Falcon VM data.
   Events, mutexes, pipes, sockets and all what's needed to run the VM in specific systems.
   System specific structures are in vm_sys_*.h, while implementation of this functions are
   in vm_sys_*.cpp.

   The structures are opaque to the virtual machine, but they can be inspected by embedding
   applications and modules (i.e. for threading support).
*/

class FALCON_DYN_CLASS SystemData
{
protected:
   /** Parent VM */
   VMachine *m_vm;

public:
   struct VM_SYS_DATA *m_sysData;

   /** Creates the system data.
      This operation is granted to succeed.
   */
   SystemData(VMachine *vm);

   /** Destroys VM specific system data.
      Will be performed only at ownwer vm's destruction.
   */
   ~SystemData();

   /** Called from VM::finalize() to stop any activity
      that may access VM when it's being freed.
    */
   void earlyCleanup();

   /** Checks wether the VM has been interrupted in a blocking wait or I/O.
      \return true if interrupted.
   */
   bool interrupted() const;

   /** Safe interthread VM interruption request on blocking I/O.
      Will work only on compliant I/O and waits.
   */
   void interrupt();

   /** Clear interruption status.
   */
   void resetInterrupt();

   /** Wait for a given count of seconds.
      \return false if interrupted.
   */
   bool sleep( numeric seconds ) const;

   /** Returns overall underlying system architecture type.
      It may be something as WIN or POSIX. More detailed informations about underlying systems
      can be retreived through modules.
      \return a static 7bit ASCII 0 terminated C string containing a very basic description
      of the underlying system overall architecture where this VM runs.
   */
   static const char *getSystemType();

   /** Send OS signals to parent VM. */
   bool becomeSignalTarget();
};

}
}

#endif

/* end of vm_sys.h */

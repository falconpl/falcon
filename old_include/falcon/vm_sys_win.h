/*
   FALCON - The Falcon Programming Language.
   FILE: vm_sys_win.h

   System specifics for the falcon VM - POSIX compliant systems.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 25 Apr 2008 17:30:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_VM_SYS_WIN_H
#define FLC_VM_SYS_WIN_H

#include <windows.h>

namespace Falcon {
namespace Sys {

struct VM_SYS_DATA
{
   HANDLE evtInterrupt;
};

}
}

#endif

/* end of vm_sys_win.h */

/*
   The Falcon Programming Language
   FILE: dynlib_sys_msvc.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - MS-Windows specific extensions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Nov 2008 21:40:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   System specific extensions.
*/

#ifndef UNICODE
#define UNICODE
#endif

#include "dynlib_sys.h"

namespace Falcon {
namespace Sys {

//=========================================
// MSVC specific code.
//
#pragma warning( disable: 4731 )  // we WANT to modify the EBP code.

void dynlib_call( void *faddress, void** parameters, int32* sizes, byte* retval )
{
   int count = 0;
   
   __asm {
      push  ebx           /* Save used registers */
      push  esi
      push  edi
      mov   edi, esp      /* Create a fake frame */
   }
   
   lbl1:
   __asm {
      mov   esi, sizes
      add   esi, count
      mov   ebx, [esi] 
      and   ebx, 0x7F      /* We're not interested in knowing the value is float */
      or    ebx, ebx       /* Check if this was the last size */
      jz    lbl2

      mov   esi,parameters /* Get next parameter */
      add   esi, count     /* Move count ptrs forward */
      mov   esi, [esi]     /* Go to the N-int buffer position */
      add   esi, ebx       /* Start from bottom */
   }
   
   lbl4:
   __asm {
      sub   esi, 4         /* Prepare to read next bytes, if we're not done... */
      mov   edx, [esi]
      push  edx            /* Push them on the stack */
      sub   ebx, 4         /* More bytes to be pushed? */
      jnz   lbl4           /* yes -- loop again */
      add   count, 4       /* Next parameter (int/ptr size fwd)... */
      jmp   lbl1
   }
   
   lbl2:
   __asm {
      call  faddress
   }

   __asm {
      mov   esi, retval
      mov   ebx, [esi]     
      or    ebx, ebx       /* No return? */
      jz    lbl100         /* Go Away */

      cmp   ebx, 0x8c      /* 0x80 + 0x0c -- long double 80 bits */
      jne   lbl5
      mov   [esi], eax
      mov   4[esi], edx
      mov   8[esi], ecx
      jmp   lbl100
   }

   lbl5:
   __asm {
      cmp   ebx, 0x88 /* 0x80 + 0x8 -- double 64 bits */
      jne   lbl6
      fstp  qword ptr [esi]
      jmp   lbl100
   }

   lbl6:
   __asm {
      cmp   ebx, 0x84 /* 0x80 + 0x4 -- float 64 bits */
      jne   lbl7
      fstp  dword ptr [esi]
      jmp   lbl100
   }
  
   lbl7:
   __asm {
      cmp   ebx, 0x08 /* */
      jne   lbl8
      mov   [esi], eax
      mov   4[esi], edx
      jmp   lbl100
   }

     
   lbl8:
   __asm {
      cmp   ebx, 0x04 /* */
      jne   lbl9
      mov   [esi], eax
      jmp   lbl100
   }

   lbl9:
   __asm {
      mov  [esi], 0 /* zero the return */
   }

   lbl100:
   __asm {
     mov esp, edi
     pop edi         /* Restore my fake frame */
     pop esi
     pop ebx
   }
}

}
}

/* end of dynlib_sys_msvc.cpp */

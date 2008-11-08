/*
   The Falcon Programming Language
   FILE: dynlib_sys_msvc.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - MS-Windows specific extensions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

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

void dynlib_void_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      mov  esi, stack_image
      mov  ecx, stack_depth
   }
   lbl1:
   __asm {
      or   ecx, ecx
      jz    lbl2
      mov   eax, [esi]
      push  eax
      add   esi,4
      dec   ecx
      jmp   lbl1
   }
   lbl2:
   __asm {
      call  faddress
      mov  esp, ebp
      pop  ebp
      ret
   }
}

void* dynlib_voidp_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      mov  esi, stack_image
      mov  ecx, stack_depth
   }
   lbl1:
   __asm {
      or   ecx, ecx
      jz    lbl2
      mov   eax, [esi]
      push  eax
      add   esi,4
      dec   ecx
      jmp   lbl1
   }
   lbl2:
   __asm {
      call  faddress
      mov  esp, ebp
      pop  ebp
      ret
   }
   return 0; // never reached
}

int32 dynlib_dword_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      mov  esi, stack_image
      mov  ecx, stack_depth
   }
   lbl1:
   __asm {
      or   ecx, ecx
      jz    lbl2
      mov   eax, [esi]
      push  eax
      add   esi,4
      dec   ecx
      jmp   lbl1
   }
   lbl2:
   __asm {
      call  faddress
      mov  esp, ebp
      pop  ebp
      ret
   }
   return 0; // never reached
}

int64 dynlib_qword_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      mov  esi, stack_image
      mov  ecx, stack_depth
   }
   lbl1:
   __asm {
      or   ecx, ecx
      jz    lbl2
      mov   eax, [esi]
      push  eax
      add   esi,4
      dec   ecx
      jmp   lbl1
   }
   lbl2:
   __asm {
      call  faddress
      mov  esp, ebp
      pop  ebp
      ret
   }
   return 0; // never reached
}

double dynlib_double_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      mov  esi, stack_image
      mov  ecx, stack_depth
   }
   lbl1:
   __asm {
      or   ecx, ecx
      jz    lbl2
      mov   eax, [esi]
      push  eax
      add   esi,4
      dec   ecx
      jmp   lbl1
   }
   lbl2:
   __asm {
      call  faddress
      mov  esp, ebp
      pop  ebp
      ret
   }
   return 0.0; // never reached
}

}
}

/* end of dynlib_sys_msvc.cpp */

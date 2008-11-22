/*
   The Falcon Programming Language
   FILE: dynlib_sys_dl.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - UNIX dload() specific extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Nov 2008 21:40:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

#include "dynlib_sys.h"
namespace Falcon {
namespace Sys {

#define dynlib_call( faddress, stack_image, stack_depth ) \
   stack_depth /= 4;\
   asm(\
      "movl 16(%ebp),%ecx\n"\
      "movl 12(%ebp),%esi\n"\
      "1: orl   %ecx, %ecx\n"\
      "jz    2f\n"\
      "movl  (%esi),%eax\n"  /* Then, transfer the stack image to the stack */\
      "pushl %eax\n"\
      "addl  $4,%esi\n"\
      "decl  %ecx\n"\
      "jmp   1b\n"\
      "2: call * 8(%ebp)\n"         /* perform the call */\
      "movl  %ebp, %esp\n"      /* Restore calling function stack frame */\
      "popl  %ebp\n"\
      "ret\n"                     /* really return */\
   );\

   // will never reach here


void dynlib_void_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
}

void* dynlib_voidp_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
   return 0; // never reached
}

int32 dynlib_dword_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
   return 0; // never reached
}

int64 dynlib_qword_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
   return 0; // never reached
}

double dynlib_double_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
   return 0.0; // never reached
}

}
}

/* end of dynlib_sys_gcc.cpp */

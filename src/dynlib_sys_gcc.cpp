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

#define dynlib_call( faddress, parameters, sizes ) \
   int count = 0; \
   __asm__ __volatile__(\
      "pushl   %%ebx \n" \
      "pushl   %%esi \n" \
      "pushl   %%edi \n" \
      "movl    %%esp, %%edi\n"      /* Create a fake frame */\
      "1: movl %2, %%esi \n" /* Get next parameter's size */ \
      "addl    %3, %%esi \n" /* Move count ptrs forward */ \
      "movl    (%%esi), %%ebx \n"\
      "andl    $0x7f, %%ebx \n"   /* We're not interested in knowing the value is float */ \
      "orl     %%ebx, %%ebx \n" /* Check if this was the last size */ \
      "jz      2f\n"\
      "movl    %1, %%esi \n" /* Get next parameter */ \
      "addl    %3, %%esi \n" /* Move count ptrs forward */ \
      "movl (%%esi), %%esi \n"  /* Go to the N-int buffero position */ \
      "4: movl (%%esi), %%edx \n"\
      "pushl   %%edx\n" /* Push them on the stack */ \
      "addl    $4,%%esi\n" /* Prepare to read next bytes, if we're not done... */ \
      "subl    $4,%%ebx\n" /* More bytes to be pushed? */ \
      "jnz     4b\n"       /* yes -- loop again */ \
      "addl    $4,%3\n"    /* Next parameter (int/ptr size fwd)... */ \
      "jmp     1b\n"\
      "2: call  *%0\n"         /* perform the call */\
      "movl  %%edi, %%esp\n"      /* Restore my fake frame */\
      "popl  %%edi\n"\
      "popl  %%esi \n" \
      "popl  %%ebx \n" \
      "movl  %%ebp, %%esp\n"      /* Restore calling function stack frame */\
      "popl  %%ebp\n"\
      "ret\n"                     /* really return */ \
      : /* no output */ \
      :"m"(faddress), "m"(parameters), "m"(sizes), "m"(count)  /* input */\
      :"%eax", "%esp"         /* clobbered register */\
   );\

   // will never reach here


void dynlib_void_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
}

void* dynlib_voidp_call( void *faddress, void** parameters, int* sizes )
{
   dynlib_call( faddress, parameters, sizes );
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

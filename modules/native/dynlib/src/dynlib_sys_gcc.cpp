/*
   The Falcon Programming Language
   FILE: dynlib_sys_gcc.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - GCC on intel32
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Nov 2008 21:40:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010 Giancarlo Niccolai

   See the LICENSE file distributed with this package for licensing details.
*/

#include "dynlib_sys.h"
namespace Falcon {
namespace Sys {



void dynlib_call( void *faddress, void** parameters, int* sizes, byte* retval )
{
   int count = 0;
// Added to support native x86-64 compilation. 1.21.2012 -gian
#ifdef __LP64__
	__asm__ __volatile__(
	"pushq	%%rbx \n"
	"pushq	%%rsi \n"
	"pushq	%%rdi \n"
	"movq	%%rsp, %%rdi\n"

	"1: movq %3, %%rsi \n"
	"addq 	%4, %%rsi \n"
	"movq 	(%%rsi), %%rbx \n"
	"andq 	$0x7f, %%rbx \n"
	"orq	%%rbx, %%rbx \n"
	"jz		2f\n"
	"movq	%2, %%rsi \n"
	"addq	%4, %%rsi \n"
	"movq	(%%rsi), %%rsi \n"
	"addq	%%rbx, %%rsi \n"
	"4: subq $4, %%rsi \n"
      "movq    (%%rsi), %%rdx \n"
      "pushq  %%rdx\n" /* Push them on the stack */
      "subq    $4,%%rbx\n" /* More bytes to be pushed? */
      "jnz     4b\n"       /* yes -- loop again */
      "addq    $4,%4\n"    /* Next parameter (int/ptr size fwd)... */
      "jmp     1b\n"

      "2: call *%1\n"         /* perform the call */

      "movq  %0, %%rsi\n"     /* Get the return format */
      "movq  (%%rsi), %%rbx\n"
      "orq   %%rbx, %%rbx\n"  /* No return? */
      "jz    100f\n"          /* Go Away */

      "cmpq  $0x8c, %%rbx\n"  /* 0x80 + 0x0c -- long double 80 bits */
      "jne   5f\n"
      "movq  %%rax, (%%rsi) \n"
      "movq  %%rdx, 4(%%rsi) \n"
      "movq  %%rcx, 8(%%rsi) \n"
      "jmp   100f \n"

      "5: cmpq  $0x88, %%rbx\n"  /* 0x80 + 0x8 -- double 64 bits */
      "jne   6f\n"
      "fstpl (%%rsi) \n"
      "jmp   100f \n"

      "6: cmpq  $0x84, %%rbx\n"  /* 0x80 + 0x4 -- float 64 bits */
      "jne   7f\n"
      "fstps (%%rsi) \n"
      "jmp   100f \n"

      "7: cmpq   $0x8, %%rbx\n"  /* long long -- 8 bytes */
      "jne   8f\n"
      "movq  %%rax, (%%rsi) \n"
      "movq  %%rdx, 4(%%rsi) \n"
      "jmp   100f \n"

      "8: cmpq  $0x4, %%rbx\n"  /* long/int/everything else -- 4 bytes */
      "jne   9f\n"
      "movq  %%rax, (%%rsi) \n"
      "jmp   100f \n"

      "9: movq $0, (%%rsi) \n"       /* zero the return */

      "100: movq  %%rdi, %%rsp\n"      /* Restore my fake frame */
      "popq %%rdi\n"
      "popq %%rsi \n"
      "popq %%rbx \n"

      :"=m"(retval) /* output */
      :"m"(faddress), "m"(parameters), "m"(sizes), "m"(count)  /* input */
      :"%rsp"         /* clobbered register */
   );


#else
   __asm__ __volatile__(
      "pushl   %%ebx \n"            /* Save used registers */
      "pushl   %%esi \n"
      "pushl   %%edi \n"
      "movl    %%esp, %%edi\n"      /* Create a fake frame */

      "1: movl %3, %%esi \n" /* Get next parameter's size */
      "addl    %4, %%esi \n" /* Move count ptrs forward */
      "movl    (%%esi), %%ebx \n"
      "andl    $0x7f, %%ebx \n"   /* We're not interested in knowing the value is float */
      "orl     %%ebx, %%ebx \n" /* Check if this was the last size */
      "jz      2f\n"
      "movl    %2, %%esi \n" /* Get next parameter */
      "addl    %4, %%esi \n" /* Move count ptrs forward */
      "movl    (%%esi), %%esi \n"  /* Go to the N-int buffero position */
      "addl    %%ebx, %%esi \n"  /* Start from bottom */
      "4: subl $4,%%esi\n" /* Prepare to read next bytes, if we're not done... */
      "movl    (%%esi), %%edx \n"
      "pushl  %%edx\n" /* Push them on the stack */
      "subl    $4,%%ebx\n" /* More bytes to be pushed? */
      "jnz     4b\n"       /* yes -- loop again */
      "addl    $4,%4\n"    /* Next parameter (int/ptr size fwd)... */
      "jmp     1b\n"

      "2: call *%1\n"         /* perform the call */

      "movl  %0, %%esi\n"     /* Get the return format */
      "movl  (%%esi), %%ebx\n"
      "orl   %%ebx, %%ebx\n"  /* No return? */
      "jz    100f\n"          /* Go Away */

      "cmpl  $0x8c, %%ebx\n"  /* 0x80 + 0x0c -- long double 80 bits */
      "jne   5f\n"
      "movl  %%eax, (%%esi) \n"
      "movl  %%edx, 4(%%esi) \n"
      "movl  %%ecx, 8(%%esi) \n"
      "jmp   100f \n"

      "5: cmpl  $0x88, %%ebx\n"  /* 0x80 + 0x8 -- double 64 bits */
      "jne   6f\n"
      "fstpl (%%esi) \n"
      "jmp   100f \n"

      "6: cmpl  $0x84, %%ebx\n"  /* 0x80 + 0x4 -- float 64 bits */
      "jne   7f\n"
      "fstps (%%esi) \n"
      "jmp   100f \n"

      "7: cmpl   $0x8, %%ebx\n"  /* long long -- 8 bytes */
      "jne   8f\n"
      "movl  %%eax, (%%esi) \n"
      "movl  %%edx, 4(%%esi) \n"
      "jmp   100f \n"

      "8: cmpl  $0x4, %%ebx\n"  /* long/int/everything else -- 4 bytes */
      "jne   9f\n"
      "movl  %%eax, (%%esi) \n"
      "jmp   100f \n"

      "9: movl $0, (%%esi) \n"       /* zero the return */

      "100: movl  %%edi, %%esp\n"      /* Restore my fake frame */
      "popl %%edi\n"
      "popl %%esi \n"
      "popl %%ebx \n"

      :"=m"(retval) /* output */
      :"m"(faddress), "m"(parameters), "m"(sizes), "m"(count)  /* input */
      :"%esp"         /* clobbered register */
   );
#endif

}

/*
 Test functions for different platforms.

static int64 _lld()
{
   return (int64) 257;
}


static float _fd()
{
   return (float) 1.5;
}


static double _d()
{
   static char buffer[sizeof(double)];
   double ldl = 1.5;
   *(double *)(buffer) = ldl;
   return ldl;
}

static long double _ld()
{
   static char buffer[sizeof(long double)];
   long double ldl = 1.5;
   *(long double *)(buffer) = ldl;
   return ldl;
}
*/
}
}

/* end of dynlib_sys_gcc.cpp */

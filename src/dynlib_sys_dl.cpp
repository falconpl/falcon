/*
   The Falcon Programming Language
   FILE: dynlib_sys_dl.cpp

   Direct dynamic library interface for Falcon
   System specific extensions - UNIX dload() specific extensions.
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

#include "dynlib_sys.h"

#include <falcon/autocstring.h>
#include <dlfcn.h>
#include <errno.h>

namespace Falcon {
namespace Sys {

void *dynlib_load( const String &libpath )
{
   AutoCString c_path( libpath );
   return dlopen( c_path.c_str(), RTLD_LAZY );
}

int dynlib_unload( void *libhandler )
{
   return dlclose(libhandler);
}

void *dynlib_get_address( void *libhandler, const String &func_name )
{
   AutoCString c_sym( func_name );
   return dlsym( libhandler, c_sym.c_str() );
}

bool dynlib_get_error( int32 &ecode, String &sError )
{
   const char *error = dlerror();
   // no error? -- don't mangle the string and return false.
   if ( error == 0 )
      return false;

   // bufferize the error and signal we have some.
   ecode = errno;
   sError.bufferize( error );
   return true;
}
#define mov_blk(src, dest, numwords) \
__asm__ __volatile__ (                                          \
                       "cld\n\t"                                \
                       "rep\n\t"                                \
                       "movsl"                                  \
                       :                                        \
                       : "S" (src), "D" (dest), "c" (numwords)  \
                       : "%ecx", "%esi", "%edi"                 \
                       )


#define dynlib_call( faddress, stack_image, stack_depth ) \
   stack_depth /= 4;\
   __asm__ __volatile__(\
      "pushl %%ebp\n"            /* first, create a temporary prolog */\
      "movl  %%esp, %%ebp\n"\
      "1: orl   %%ecx, %%ecx\n"\
      "jz    2f\n"\
      "movl  (%%esi),%%eax\n"  /* Then, transfer the stack image to the stack */\
      "pushl %%eax\n"\
      "addl  $4,%%esi\n"\
      "decl  %%ecx\n"\
      "jmp   1b\n"\
      "2: call  *%%edx\n"         /* perform the call */\
      "movl  %%ebp, %%esp\n"      /* Restore temporary stack frame */\
      "popl  %%ebp\n"\
      "movl  %%ebp, %%esp\n"      /* Restore calling function stack frame */\
      "popl  %%ebp\n"\
      "ret\n"                     /* really return */\
      : /* no output */\
      :"d"(faddress), "S"(stack_image), "c"(stack_depth)  /* input */\
      :"%eax", "%ebp", "%esp"         /* clobbered register */\
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

uint32 dynlib_dword_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   dynlib_call( faddress, stack_image, stack_depth );
   return 0; // never reached
}

uint64 dynlib_qword_call( void *faddress, byte *stack_image, uint32 stack_depth )
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

/* end of dynlib_sys_dl.cpp */

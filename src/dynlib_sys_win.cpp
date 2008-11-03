/*
   The Falcon Programming Language
   FILE: dynlib_sys_win.cpp

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
#include "dynlib_st.h"
#include <windows.h>
#include <falcon/autowstring.h>
#include <falcon/autocstring.h>

#include <stdio.h>
namespace Falcon {
namespace Sys {

void *dynlib_load( const String &libpath )
{
   // we must convert falcon path into windows path
   String lpath = libpath;
   for( uint32 i = 0; i < lpath.length(); i++ )
      if ( lpath.getCharAt(i) == '/' )
         lpath.setCharAt(i, '\\' );

   AutoWString wstr( lpath );
   return (void *) LoadLibraryW( wstr.w_str() );
}


int dynlib_unload( void *libhandler )
{
   HMODULE handle = (HMODULE) libhandler;
   FreeLibrary( handle );
   return 0;
}


void *dynlib_get_address( void *libhandler, const String &func_name )
{
   HMODULE handle = (HMODULE) libhandler;
   AutoCString sym( func_name );
   return (void *) GetProcAddress( handle, sym.c_str() );
}


bool dynlib_get_error( int32 &ecode, String &sError )
{
   DWORD nError = GetLastError();
   if( nError == 0 )
   {
      return false;
   }

   ecode = (int32) nError;
   
   LPWSTR pBuffer = NULL;
   if ( FormatMessage(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER,   //flags
            0, // lpsource
            nError, //msgid
            0, // default language
            (LPWSTR)&pBuffer,  // output buffer
            0, // size - ignored
            0  // args - ignored
            ) == 0
         )
   {
      // failed to get the message
      sError = "";
      return true;
   }

   // Already in utf-16 format (safe), so we can convert.
   sError.bufferize( pBuffer );
   LocalFree( pBuffer );
   return true;
}


//=========================================
// MSVC specific code.
//
#pragma warning( disable: 4731 )  // we WANT to modify the EBP code.

void dynlib_void_call( void *faddress, byte *stack_image, uint32 stack_depth )
{
   stack_depth /= 4;
   __asm {
      lea   esi, stack_image
      mov   ecx, stack_depth
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
      lea   esi, stack_image
      mov   ecx, stack_depth
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
      lea   esi, stack_image
      mov   ecx, stack_depth
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
      lea   esi, stack_image
      mov   ecx, stack_depth
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

/* end of dynlib_sys_win.cpp */

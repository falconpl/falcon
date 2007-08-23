/*
   FALCON - Falcon advanced simple text evaluator.
   FILE: dll_win.cpp
   $Id: dll_win.cpp,v 1.2 2006/11/19 11:25:39 gian Exp $

   Implementation of windows specific DLL system
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/


// Must include also windows.h
#include <falcon/dll_win.h>
#include <falcon/memory.h>
#include <falcon/sys.h>

namespace Falcon
{

DllLoader_win::~DllLoader_win()
{
   close();
}

bool DllLoader_win::open( const String &dll_name_fal )
{
   if( m_module != NULL )
      if ( ! FreeLibrary( m_module ) )
         return false;

	String dll_name;
	Sys::falconToWin_fname( dll_name_fal, "", dll_name );

	uint32 bufsize = dll_name.length() * sizeof( wchar_t ) + sizeof( wchar_t );
	wchar_t *dll_name_wc = (wchar_t *) memAlloc( bufsize );
   dll_name.toWideString( dll_name_wc, bufsize );

   m_module = LoadLibraryW( dll_name_wc );

   if ( m_module == NULL ) {
      m_error = GetLastError();
      if (  m_error  == ERROR_CALL_NOT_IMPLEMENTED )
      {
         char *dll_name_c = (char *) dll_name_wc;
         if( dll_name.toCString( dll_name_c, bufsize ) > 0 )
            m_module = LoadLibrary( dll_name_c );
      }
   }

   memFree( dll_name_wc );

   if ( m_module == NULL )
   {
      m_error = GetLastError();
      return false;
   }

   return true;
}

bool DllLoader_win::close()
{
   if ( m_module != NULL ) {
      if ( FreeLibrary( m_module ) ) {
         m_module = NULL;
         return true;
      }
   }
   return false;
}

void DllLoader_win::assign( DllLoader_win &other )
{
   if ( m_module != NULL )
      close();

   m_module = other.m_module;
   other.m_module = NULL;
}


DllFunc DllLoader_win::getSymbol( const char *sym_name ) const
{
   if ( m_module != NULL )
      return DllFunc( (void*)GetProcAddress( m_module, (LPCSTR) sym_name ) );
   return DllFunc( 0 );
}

bool DllLoader_win::isDllMark( char ch1, char ch2 )
{
   if ( ch1 == 'M' && ch2 == 'Z' ) return true;
   return false;
}

void DllLoader_win::getErrorDescription( String &descr ) const
{
   LPVOID lpMsgBuf;

   DWORD res = FormatMessage(
      FORMAT_MESSAGE_ALLOCATE_BUFFER |
      FORMAT_MESSAGE_FROM_SYSTEM,
      0,
      m_error,
      LANG_USER_DEFAULT,
      (LPTSTR) &lpMsgBuf,
      0,
      NULL
    );

   if ( res == 0 ) {
      descr = "Impossible to retreive error description";
   }
   else
   {
      descr = (char *) lpMsgBuf;
      // force to copy
      descr.reserve( 0 );
   }

   LocalFree(lpMsgBuf);
}


}


/* end of dll_win.cpp */

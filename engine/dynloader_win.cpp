/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader_win.cpp

   Native shared object based module loader -- Ms Windows standard ext
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader_win.cpp"

#include <falcon/dynloader.h>
#include <falcon/string.h>
#include <falcon/path.h>
#include <falcon/module.h>
#include <falcon/stderrors.h>
#include <falcon/autocstring.h>

#include <windows.h>


namespace Falcon
{


void DynLibrary::open_sys(const String& path)
{
   String dll_name = path;
   Path::uriToWin( dll_name );

   uint32 bufsize = dll_name.length() * sizeof( wchar_t ) + sizeof( wchar_t );
   wchar_t *dll_name_wc = (wchar_t *) malloc( bufsize );
   dll_name.toWideString( dll_name_wc, bufsize );

   HMODULE module = LoadLibraryW( dll_name_wc );
   DWORD error;

   if ( module == NULL )
   {
      error = GetLastError();
      if (  error  == ERROR_CALL_NOT_IMPLEMENTED )
      {
         char *dll_name_c = (char *) dll_name_wc;
         if( dll_name.toCString( dll_name_c, bufsize ) > 0 )
            module = LoadLibrary( dll_name_c );
      }
   }

   free( dll_name_wc );

   if ( module == NULL )
   {
      throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
                         .origin( ErrorParam::e_orig_loader )
                         .sysError( GetLastError() )
                         .extra( path ) );
   }

   m_sysData = module;
}


void DynLibrary::close_sys()
{
   if( m_sysData != 0 )
   {
      FreeLibrary( (HMODULE) m_sysData );
      m_sysData = 0;
   }
}


void* DynLibrary::getDynSymbol_nothrow( const String& str_symname ) const
{
   AutoCString sn(str_symname);
   const char* symname = sn.c_str();
   void* sym = (void*)GetProcAddress( (HINSTANCE)m_sysData, symname );
   return sym;
}

const String& DynLoader::sysExtension()
{
   static String ext = "dll";
   return ext;
}

}

/* end of dynloader_win.cpp */

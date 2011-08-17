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
#include <falcon/dynunloader.h>
#include <falcon/string.h>
#include <falcon/path.h>
#include <falcon/module.h>
#include <falcon/ioerror.h>
#include <falcon/memory.h>

#include <windows.h>



namespace Falcon
{

Module* DynLoader::load_sys( const String& filePath )
{
   String dll_name = filePath;
   Path::uriToWin( dll_name );

   uint32 bufsize = dll_name.length() * sizeof( wchar_t ) + sizeof( wchar_t );
   wchar_t *dll_name_wc = (wchar_t *) memAlloc( bufsize );
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

   memFree( dll_name_wc );

   if ( module == NULL )
   {
      throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
                         .origin( ErrorParam::e_orig_loader )
                         .sysError( GetLastError() )
                         .extra( filePath ) );
      return false;
   }

   typedef Module* (*module_init_type)();
   module_init_type module_init;

   module_init = (module_init_type) GetProcAddress( module, DEFALUT_FALCON_MODULE_INIT_NAME );

   Module* mod = module_init();
   if( mod == 0 )
   {
      throw new IOError( ErrorParam( e_bininit, __LINE__, SRC )
                         .origin( ErrorParam::e_orig_loader )
                         .extra( filePath ) );
   }

   mod->setDynUnloader( new DynUnloader( module ) );
   return mod;
}

const String& DynLoader::sysExtension()
{
   static String ext = ".dll";
   return ext;
}

//============================================================
//

void DynUnloader::unload()
{
   if( m_sysData != 0 )
   {
      FreeLibrary( (HMODULE) m_sysData );
      m_sysData = 0;
   }
}

}

/* end of dynloader_win.cpp */

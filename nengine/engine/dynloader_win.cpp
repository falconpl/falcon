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

#include <windows.h>
#ifdef UNICODE // defined in windows.h
    #include <falcon/autowstring.h>
#else
    #include <falcon/autocstring.h>
#endif


namespace Falcon
{

Module* DynLoader::load_sys( const String& filePath )
{
   String winFilePath = filePath;
   Path::uriToWin( winFilePath );

#ifdef UNICODE
   AutoWString cpath( winFilePath );
   const wchar_t* rawPath = cpath.w_str();
#else
   AutoCString cpath( winFilePath );
   const char* rawPath = cpath.c_str();
#endif

   HMODULE module = LoadLibrary( rawPath );
   if( module == 0 )
   {
       throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
          .origin( ErrorParam::e_orig_loader )
          .sysError( GetLastError() )
          .extra( filePath ) );
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

/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader_posix.cpp

   Native shared object based module loader -- POSIX standard ext.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader_posix.cpp"

#include <falcon/dynloader.h>
#include <falcon/dynunloader.h>
#include <falcon/stderrors.h>
#include <falcon/autocstring.h>
#include <falcon/string.h>

#include <dlfcn.h>
#include <errno.h>

#include <falcon/module.h>

namespace Falcon
{
   
Module* DynLoader::load_sys( const String& filePath )
{
   AutoCString cname( filePath );
   errno = 0;
   void* modData = ::dlopen( cname.c_str(), RTLD_NOW |RTLD_LOCAL);
   if( modData == 0 )
   {
      throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
         .origin( ErrorParam::e_orig_loader )
         .sysError(errno)
         .extra( filePath + " - " + dlerror() ) );
   }
 
   Module* (*module_init)();
   (void) dlerror();
   module_init = (Module* (*)())::dlsym( modData, DEFALUT_FALCON_MODULE_INIT_NAME );
   if ( module_init == 0 )
   {
      throw new IOError( ErrorParam( e_binstartup, __LINE__, SRC )
         .origin( ErrorParam::e_orig_loader )
         .extra( filePath + " - " + dlerror() ) );
   }
   Module* mod = module_init();
   
   if( mod == 0 )
   {
      throw new IOError( ErrorParam( e_bininit, __LINE__, SRC )
         .origin( ErrorParam::e_orig_loader )
         .extra( filePath ) );
   }
   
   mod->setDynUnloader( new DynUnloader( modData ) );
   return mod;
}


const String& DynLoader::sysExtension()
{
   static String ext("so");
   return ext;
}

//============================================================
//

void DynUnloader::unload()
{
   if( m_sysData != 0 )
   {
      dlclose( m_sysData );
      m_sysData = 0;
   }
}

}

/* end of dynloader_posix.cpp */

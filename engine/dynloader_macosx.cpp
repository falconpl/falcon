/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader_macosx.cpp

   Native shared object based module loader -- MacOSx standard ext.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader_macosx.cpp"

#include <falcon/dynloader.h>
#include <falcon/string.h>

#include <falcon/stderrors.h>
#include <falcon/autocstring.h>
#include <falcon/string.h>
#include <dlfcn.h>
#include <errno.h>
#include <falcon/module.h>

#include <dlfcn.h>

namespace Falcon
{

void DynLibrary::open_sys(const String& path)
{
   AutoCString cname( path );
   errno = 0;
   void* modData = ::dlopen( cname.c_str(), RTLD_NOW |RTLD_LOCAL);
   if( modData == 0 )
   {
      throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
         .origin( ErrorParam::e_orig_loader )
         .sysError( (uint32) errno )
         .extra( path + " - " + dlerror() ) );
   }

   m_sysData = modData;
}


void DynLibrary::close_sys()
{
   int res = ::dlclose( m_sysData );
   if( res != 0 )
   {
      throw new IOError( ErrorParam( e_binload, __LINE__, SRC )
               .origin( ErrorParam::e_orig_loader )
               .sysError( (uint32) errno )
               .extra( dlerror() )
               );
   }

   m_sysData = 0;
}


void* DynLibrary::getDynSymbol_nothrow( const String& str_symname ) const
{
   AutoCString sn(str_symname);
   const char* symname = sn.c_str();
   void* sym = ::dlsym( m_sysData, symname );
   return sym;
}


const String& DynLoader::sysExtension()
{
   static String ext = ".dylib";
   return ext;
}

}

/* end of dynloader_macosx.cpp */

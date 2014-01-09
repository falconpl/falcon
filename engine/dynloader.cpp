/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader.cpp

   Native shared object based module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader.cpp"

#include <falcon/dynloader.h>
#include <falcon/module.h>
#include <falcon/fassert.h>
#include <falcon/stderrors.h>

#include <stdio.h>

namespace Falcon
{

//===================================================================
// DynLibrary
//

DynLibrary::DynLibrary():
   m_sysData(0)
{}


DynLibrary::DynLibrary( const String& path ):
   m_sysData(0)
{
   open(path);
}


DynLibrary::~DynLibrary()
{
   this->close();
}


void DynLibrary::open(const String& path)
{
   if( m_sysData != 0 )
   {
      return;
   }
   m_path = path;
   open_sys(path);
}


void* DynLibrary::getDynSymbol( const String& symname ) const
{
   void* sym = getDynSymbol_nothrow( symname );

   if( sym == 0 )
   {
      throw FALCON_SIGN_XERROR( AccessError, e_undef_sym, .extra(symname + " in " + m_path).origin(ErrorParam::e_orig_loader) );
   }

   return sym;
}


void DynLibrary::close()
{
   if( m_sysData == 0 )
   {
      return;
   }

   close_sys();

   m_sysData = 0;
}

//===================================================================
// DynLoader
//

DynLoader::DynLoader()
{}

DynLoader::~DynLoader()
{}
   
Module* DynLoader::load( const String& modpath, const String& modname )
{
   DynLibrary* dl = new DynLibrary(modpath);

   Module* (*module_init)();
   module_init = (Module* (*)()) dl->getDynSymbol(DEFALUT_FALCON_MODULE_INIT_NAME);
   Module* mod = module_init();

   if( mod == 0 )
   {
      throw new IOError( ErrorParam( e_bininit, __LINE__, SRC )
         .origin( ErrorParam::e_orig_loader )
         .extra( modpath ) );
   }

   mod->setDynUnloader( dl );

   mod->uri( modpath );
   mod->name( modname );
   return mod;
}

}

/* end of dynloader.cpp */

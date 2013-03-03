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
#include <falcon/dynunloader.h>
#include <falcon/module.h>
#include <falcon/fassert.h>

namespace Falcon
{

DynLoader::DynLoader()
{}

DynLoader::~DynLoader()
{}
   
Module* DynLoader::load( const String& modpath, const String& modname )
{   
   Module* mod = load_sys( modpath );
   fassert( mod != 0 ); // should throw on problem.
   mod->uri( modpath );
   mod->name( modname );
   return mod;
}


//===================================================================
//

DynUnloader::DynUnloader( void* sysData ):
   m_sysData(sysData)
{}

DynUnloader::~DynUnloader()
{
   if( m_sysData != 0 )
   {
      unload();
   }
   m_sysData = 0;
}
 
}

/* end of dynloader.cpp */

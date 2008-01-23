/*
   FALCON - The Falcon Programming Language.
   FILE: flc_runtime.cpp
   $Id: runtime.cpp,v 1.11 2007/08/11 19:02:32 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer ago 18 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/module.h>
#include <falcon/runtime.h>
#include <falcon/traits.h>
#include <falcon/vm.h>

namespace Falcon {

ModuleMap::ModuleMap():
   Map( &traits::t_stringptr, &traits::t_voidp )
{
}

ModuleVector::ModuleVector():
	GenericVector( &traits::t_voidp )
{
}

Runtime::Runtime():
   m_loader( 0 ),
   m_provider( 0 )
{}

Runtime::Runtime( ModuleLoader *loader, VMachine *prov ):
   m_loader( loader ),
   m_provider( prov )
{}

/** Declared here to avoid inlining of destructor. */
Runtime::~Runtime()
{
   for ( uint32 i = 0; i < m_modvect.size(); i++ )
   {
      m_modvect.moduleAt( i )->decref();
   }
}


bool Runtime::addModule( Module *mod )
{
   if ( m_modules.find( &mod->name() ) != 0 )
      return true;  // already in..

   m_modules.insert( &mod->name(), mod );

   if ( m_loader != 0 && ! mod->dependencies().empty() )
   {
      ListElement *deps = mod->dependencies().begin();
      while( deps != 0 )
      {
         const String *moduleName = (const String *) deps->data();

         // if we have a provider, skip this module if already found VM
         if( (m_provider != 0 && m_provider->findModule( *moduleName ) != 0)  ||
            // ... or do we have already loaded the module?
            m_modules.find( moduleName ) != 0 )
         {
            // already in
            deps = deps->next();
            continue;
         }

         Module *l;
         if ( (l = m_loader->loadName( *moduleName, mod->name() )) == 0)
            return false;

         if ( ! addModule( l ) )
         {
            l->decref();
            return false;
         }
         l->decref();

         deps = deps->next();
      }
   }

   m_modvect.push( mod );
   mod->incref();

   return true;
}

bool Runtime::loadName( const String &name, const String &parent )
{
   Module *l;
   if ( (l = m_loader->loadName( name, parent )) == 0)
      return false;

   bool ret = addModule( l );
   l->decref();
   return ret;
}

bool Runtime::loadFile( const String &file )
{
   Module *l;
   if ( (l = m_loader->loadFile( file )) == 0)
      return false;

   bool ret = addModule( l );
   l->decref();
   return ret;
}

}

/* end of flc_runtime.cpp */

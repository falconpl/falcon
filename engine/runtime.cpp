/*
   FALCON - The Falcon Programming Language.
   FILE: flc_runtime.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer ago 18 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
   m_provider( 0 ),
   m_modPending( &traits::t_stringptr, &traits::t_int )
{}

Runtime::Runtime( ModuleLoader *loader, VMachine *prov ):
   m_loader( loader ),
   m_provider( prov ),
   m_modPending( &traits::t_stringptr, &traits::t_int )
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

      m_modvect.push( mod );
   }
   else
   {
      int insertAt = m_modvect.size();

      // re-sort the array if we had some previous dependency.
      int *pending = (int *) m_modPending.find( &mod->name() );
      if ( pending != 0 )
      {
         insertAt = *pending;
         m_modvect.insert( mod, (uint32) insertAt );
      }
      else
         m_modvect.push( mod );

      // then, record pending modules for THIS module, if any.
      ListElement *deps = mod->dependencies().begin();
      while( deps != 0 )
      {
         const String *moduleName = (const String *) deps->data();

         // if the module is missing both from our module list and dependency list
         // add a dependency here
         if ( m_modules.find( moduleName ) == 0 && m_modPending.find( moduleName ) == 0 )
         {
            m_modPending.insert( moduleName, &insertAt );
         }
         deps = deps->next();
      }
   }


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

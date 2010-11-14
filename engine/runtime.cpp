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
#include <falcon/path.h>

namespace Falcon {

ModuleMap::ModuleMap():
   Map( &traits::t_stringptr(), &traits::t_voidp() )
{
}

ModuleVector::ModuleVector():
	GenericVector( &traits::t_voidp() )
{
}

Runtime::Runtime():
   m_loader( 0 ),
   m_provider( 0 ),
   m_modPending( &traits::t_stringptr(), &traits::t_int() ),
   m_hasMainModule( true )
{}

Runtime::Runtime( ModuleLoader *loader, VMachine *prov ):
   m_loader( loader ),
   m_provider( prov ),
   m_modPending( &traits::t_stringptr(), &traits::t_int() ),
   m_hasMainModule( true )
{}

/** Declared here to avoid inlining of destructor. */
Runtime::~Runtime()
{
   for ( uint32 i = 0; i < m_modvect.size(); i++ )
   {
      delete m_modvect.moduleDepAt( i );
   }
}


void Runtime::addModule( Module *mod, bool isPrivate )
{
   if ( m_modules.find( &mod->name() ) != 0 )
      return;  // already in..

   ModuleDep *dep = new ModuleDep( mod, isPrivate );

   m_modules.insert( &mod->name(), dep );

   if ( m_loader != 0 && ! mod->dependencies().empty() )
   {
      MapIterator deps = mod->dependencies().begin();
      while( deps.hasCurrent() )
      {
         const ModuleDepData *depdata = *(const ModuleDepData **) deps.currentValue();
         const String &moduleName = depdata->moduleName();

         // if we have a provider, skip this module if already found VM
         LiveModule *livemod;
         ModuleDep **olddep;
         if( m_provider != 0 && (livemod = m_provider->findModule( moduleName )) != 0 )
         {
            if ( livemod->isPrivate() && ! depdata->isPrivate() )
            {
               // just insert the module with the request to extend its privacy.
               m_modvect.push( new ModuleDep( const_cast<Module *>(livemod->module()), true ) );
            }

            // anyhow, we don't need to perform another load.
            deps.next();
            continue;
         }
         else {
            // ... or do we have already loaded the module?
            if( (olddep = (ModuleDep **) m_modules.find( &moduleName )) != 0 )
            {
               // already in? -- should we broaden the publishing?
               if( (*olddep)->isPrivate() && ! depdata->isPrivate() )
               {
                  (*olddep)->setPrivate( false );
               }
               // anyhow, we don't need to perform another load.
               deps.next();
               continue;
            }
         }

         Module *l = 0;
         try {
            if( depdata->isFile() )
            {
               // if the path is relative, then it's relative to the parent module path.
               Path p(moduleName);
               if ( !p.isAbsolute() )
               {
                 l = m_loader->loadFile(  Path(mod->path()).getFullLocation() + "/" + moduleName );
               }
               else
               {
                 l = m_loader->loadFile( moduleName );
               }
            }
            else
            {
               l = m_loader->loadName( moduleName, mod->name() );
            }
         }
         catch( Error* e)
         {
            if ( e->module() == "" )
               e->module( moduleName );
            m_modules.erase( &mod->name() );
            delete dep;
            CodeError* ce = new CodeError(
                  ErrorParam( e_loaderror )
                  .module( mod->name() )
                  .extra( "loading " + moduleName )
                  .origin( e_orig_loader ) );
            ce->appendSubError( e );
            e->decref();
            throw ce;
         }

         fassert( l != 0 );

         try
         {
            addModule( l, depdata->isPrivate() );
         }
         catch( Error* )
         {
            l->decref();
            m_modules.erase( &mod->name() );
            delete dep;
            throw;
         }
         l->decref();

         deps.next();
      }

      m_modvect.push( dep );
   }
   else
   {
      int insertAt = m_modvect.size();

      // re-sort the array if we had some previous dependency.
      int *pending = (int *) m_modPending.find( &mod->name() );
      if ( pending != 0 )
      {
         insertAt = *pending;
         m_modvect.insert( dep, (uint32) insertAt );
      }
      else
         m_modvect.push( dep );

      // then, record pending modules for THIS module, if any.
      MapIterator deps = mod->dependencies().begin();
      while( deps.hasCurrent() )
      {
         const ModuleDepData *depdata = *(const ModuleDepData **) deps.currentValue();
         const String &moduleName = depdata->moduleName();

         // if the module is missing both from our module list and dependency list
         // add a dependency here
         if ( m_modules.find( &moduleName ) == 0 && m_modPending.find( &moduleName ) == 0 )
         {
            m_modPending.insert( &moduleName, &insertAt );
         }
         deps.next();
      }
   }
}

void Runtime::loadName( const String &name, const String &parent, bool bIsPrivate )
{
   Module *l = m_loader->loadName( name, parent );

   try
   {
      addModule( l, bIsPrivate );
      l->decref();
   }
   catch( Error* )
   {
      l->decref();
      throw;
   }
}

void Runtime::loadFile( const String &file, bool bIsPrivate )
{
   Module *l = m_loader->loadFile( file, ModuleLoader::t_none, true );

   try
   {
   	if( bIsPrivate )
   	{
   		// mangle the file name...
   		String name = l->name();
   		l->name( name.A("-").N(rand()).N(rand()));
   	}

      addModule( l, bIsPrivate );
      l->decref();
   }
   catch( Error* )
   {
      l->decref();
      throw;
   }
}

}

/* end of flc_runtime.cpp */

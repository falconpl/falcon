/*
   FALCON - The Falcon Programming Language.
   FILE: modgroup.cpp

   Group of modules involved in a single link operation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/modgroup.cpp"

#include <falcon/modgroup.h>
#include <falcon/trace.h>
#include <falcon/error.h>
#include <falcon/linkerror.h>
#include <falcon/genericerror.h>
#include <falcon/modspace.h>
#include <falcon/module.h>
#include <falcon/symbol.h>
#include <falcon/vmcontext.h>

#include "modmap_private.h"
#include "symbolmap_private.h"

#include <deque>

namespace Falcon 
{

class ModGroup::Private
{
public:
   typedef std::deque<Module*> ModList;
   ModList m_loadOrder;
   
   typedef std::deque<Error*> ErrorList;
   ErrorList m_errors;

   
   Private()
   {}
   
   ~Private()
   {
      clearErrors();
   }
   
   
   void clearErrors()
   {
      ErrorList::iterator iter = m_errors.begin();
      while( iter != m_errors.end() )
      {
         (*iter)->decref();
         ++iter;
      }
      m_errors.clear();
   }
};


ModGroup::ModGroup( ModSpace* owner ):
   _p( new Private ),
   m_owner( owner )
{
}


ModGroup::~ModGroup()
{
   delete _p;
}


bool ModGroup::add( Module* mod, t_loadMode mode )
{
   TRACE( "ModGroup::add %s with mode %s",
      mod->name().c_ize(), mode == e_lm_load ? "load" : 
            (mode == e_lm_import_public ? "import public" 
                     : "import private" ) 
      );
   
   // already added?
   ModMap::Entry* entry = m_modules.findByModule( mod );
   if( entry != 0 )
   {
      // should we promote it?
      if ( mode == e_lm_load 
           || (mode == e_lm_import_public && entry->imode() != e_lm_load) )
      {
         entry->imode( mode );
      }
      // we don't have to resolve its static requirements, as it's already added.
   }
   else
   {
      m_modules.add( mod, mode );
      _p->m_loadOrder.push_back( mod );
      
      // we're the grop owning the module!
      mod->moduleGroup( this );
            
      // now that we know each other, we can resolve the module dependencies.
      mod->resolveStaticReqs();
   }
   
   return true;
}
  

void ModGroup::addLinkError( int err_id, const String& modName, const Symbol* sym, const String& extra )
{
   TRACE( "ModGroup::addLinkError %d -- %s", err_id, modName.c_ize() );
   
   Error* e = new LinkError( ErrorParam( err_id )
      .origin(ErrorParam::e_orig_linker)
      .line( sym != 0 ? sym->declaredAt() : 0 )
      .extra( extra )
      .module( modName != "" ? modName : "<internal>" ) 
      .symbol( sym != 0 ? sym->name() : "" ));
      
   addLinkError( e );
   e->decref();
}


void ModGroup::addLinkError( Error* e )
{
   TRACE( "ModGroup::addLinkError %d -- %s", e->errorCode(), e->module().c_ize() );
   _p->m_errors.push_back( e );
   e->incref();
}


Error* ModGroup::makeError() const
{
   Error* e = new LinkError( ErrorParam( e_link_error, __LINE__, SRC )
      .origin(ErrorParam::e_orig_linker) );
   
   Private::ErrorList::iterator iter = _p->m_errors.begin();
   while( iter != _p->m_errors.end() )
   {
      e->appendSubError( *iter );
      ++iter;
   }
   
   _p->clearErrors();
   return e;
}

   
bool ModGroup::link()
{
   TRACE( "ModGroup::link started with %d modules.", modules()._p->m_modMap.size() );
   _p->clearErrors();
   
   // In this function we access modmap without locks,
   // but locks are actually meant for ModSpace access, which may be MT,
   // while ModGroup access is always thread-local.
   
   // If we can fulfill direct imports...
   bool bSuccess = linkDirectImports();
   // ... and we can fulfill export requests...
   bSuccess = linkExports( bSuccess );   
   // ... and then we can also fulfil indirect import requests ... 
   bSuccess = linkGenericImports( bSuccess );
      
   // if we succeded in all the steps then we can commit our modules.
   // (Notice that linkExports and linkGenericImports have logic to return false
   // -- if the parameter was false).      
   commitModules(bSuccess);
   
   if( bSuccess )
   {
      TRACE( "ModGroup::link success -- now left with %d modules.", 
         modules()._p->m_modMap.size() );
   }
   else
   {
      MESSAGE( "ModGroup::link failed." );
   }
   
   return bSuccess;
}
 

bool ModGroup::linkDirectImports()
{
   MESSAGE1( "ModGroup::linkDirectImports begin.");
   ModMap::Private::ModEntryMap& modMap = m_modules._p->m_modMap;   
   ModMap::Private::ModEntryMap::const_iterator iter = modMap.begin();
   
   while( iter != modMap.end() )
   {
      ModMap::Entry* entry = iter->second;
      Module* mod = entry->module();
      mod->resolveDirectImports( true );      
      ++iter;
   }
   
   TRACE1( "ModGroup::linkDirectImports %s.", 
               _p->m_errors.empty() ? "success" : "failure" );
   return _p->m_errors.empty();
}


bool ModGroup::linkExports( bool bForReal )
{
   TRACE1( "ModGroup::linkExports begin %s.", bForReal ? "real" : "dry");
   ModMap::Private::ModEntryMap& modMap = m_modules._p->m_modMap;   
   ModMap::Private::ModEntryMap::const_iterator iter = modMap.begin();

   while( iter != modMap.end() )
   {
      ModMap::Entry* entry = iter->second;
      Module* mod = entry->module();
      
      // Export symbols in the group
      if( entry->imode() == e_lm_load )
      {
         // this also creates a token for each exported symbol on the space.
         mod->exportSymsInGroup( bForReal );
      }
      
      ++iter;
   }
   
   TRACE1( "ModGroup::linkExports %s.", 
               _p->m_errors.empty() ? "success" : "failure" );
   return bForReal && _p->m_errors.empty();
}


bool ModGroup::linkGenericImports( bool bForReal )
{
   TRACE1( "ModGroup::linkGenericImports begin %s.", bForReal ? "real" : "dry");

   ModMap::Private::ModEntryMap& modMap = m_modules._p->m_modMap;   
   ModMap::Private::ModEntryMap::const_iterator iter = modMap.begin();

   while( iter != modMap.end() )
   {
      ModMap::Entry* entry = iter->second;
      Module* mod = entry->module();            
      mod->resolveGenericImports( bForReal );
      ++iter;
   }
   
   TRACE1( "ModGroup::linkGenericImports %s.", 
               _p->m_errors.empty() ? "success" : "failure" );
   return bForReal && _p->m_errors.empty();
}


void ModGroup::commitModules( bool bForReal )
{
   TRACE1( "ModGroup::commitModules begin %s.", bForReal ? "real" : "dry" );
   ModMap::Private::ModEntryMap& modMap = m_modules._p->m_modMap;   
   ModMap::Private::ModEntryMap::iterator iter = modMap.begin();

   while( iter != modMap.end() )
   {
      ModMap::Entry* entry = iter->second;
      Module* mod = entry->module();
      
      // Transfer the module to the space, if required.
      if( entry->imode() != e_lm_import_private )
      {
         if( bForReal )
         {
            // the module has not a group anymore.
            mod->moduleGroup(0);
            // Send the group to the owner.
            m_owner->modules().add( mod, entry->imode() );
            // remove the transferred module
            ModMap::Private::ModEntryMap::iterator old = iter;
            ++iter;
            modMap.erase( old );
         }
         else
         {
            ++iter;
         }
      }
      else
      {
         ++iter;
      }
   }
   
   MESSAGE1( "ModGroup::commitModules committing symbols." );
   
   // and transfer all the exported symbols.
   SymbolMap::Private::SymModMap& symMap = m_symbols._p->m_syms;   
   SymbolMap::Private::SymModMap::const_iterator siter = symMap.begin();

   while( siter != symMap.end() )
   {
      SymbolMap::Entry* entry = siter->second;
      if( bForReal )
      {
         m_owner->symbols().add( entry->symbol(), entry->module() );
      }
      
      m_owner->removeSymbolToken( entry->symbol()->name(), entry->module()->name() );
      
      ++siter;
   }
   
   MESSAGE1( "ModGroup::commitModules end." );
   // we transferred all the symbols
   m_symbols._p->m_syms.clear();
}


void ModGroup::readyVM( VMContext* ctx )
{
   Private::ModList& mods = _p->m_loadOrder;
   
   // insertion goes first to last because execution goes last to first.   
   Private::ModList::const_iterator imods = mods.begin();
   while( imods != mods.end() )
   {
      const Module* mod = *imods;
      // TODO -- invoke the init methods.
      
      // TODO -- specific space for Main.
      Function* main = mod->getFunction("__main__");
      if( main != 0 )
      {
         ctx->call( main, 0 );
      }
            
      ++imods;
   }
}


}

/* end of modgroup.cpp */

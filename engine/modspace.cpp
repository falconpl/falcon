/*
   FALCON - The Falcon Programming Language.
   FILE: modspace.cpp

   Module space for the Falcon virtual machine
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 14:27:51 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/trace.h>
#include <falcon/modspace.h>
#include <falcon/error.h>
#include <falcon/symbol.h>
#include <falcon/module.h>
#include <falcon/vmcontext.h>
#include <falcon/modloader.h>
#include <falcon/errors/linkerror.h>
#include <falcon/itemarray.h>
#include <falcon/errors/genericerror.h>

#include <map>
#include <deque>

#include "module_private.h"

namespace Falcon {

class ModSpace::ModuleData
{
public:
   Module* m_mod;
   bool m_bExport;
   bool m_bExported;
   bool m_bOwn;
   
   ModuleData( Module* mod, bool bExport, bool bOwn ):
      m_mod( mod ),
      m_bExport( bExport ),
      m_bExported( false ),
      m_bOwn( bOwn )
   {      
   }
   
   ~ModuleData()
   {
      if ( m_bOwn )
      {
         delete m_mod;
      }
   }
};
   

class ModSpace::Private
{
public:
      
   typedef std::deque<ModSpace::ModuleData*> ModList;
   ModList m_invokeOrder;
   ModList m_linkOrder;
   
   class ExportSymEntry{
   public:
      Module* m_mod;
      Symbol* m_sym;
      
      ExportSymEntry(): m_mod(0), m_sym(0) {}
      
      ExportSymEntry(Module* mod, Symbol* sym ):
         m_mod(mod),
         m_sym(sym)
      {}
      
      ExportSymEntry( const ExportSymEntry& other ):
         m_mod(other.m_mod),
         m_sym(other.m_sym)
      {}
   };
   
   typedef std::map<String, ExportSymEntry> ExportSymMap;
   ExportSymMap m_symMap;
   
   typedef std::map<String, ModSpace::ModuleData*> ModMap;
   ModMap m_modmap;
   ModMap m_modmapByUri;
   
   ItemArray m_values;
   
   Private()
   {}
   
   ~Private()
   {
      // use the modmap, because modmapByUri can be incomplete (theoretically)
      ModMap::iterator iter = m_modmap.begin();
      while( iter != m_modmap.end() )
      {
         delete iter->second;
         ++iter;
      }
   }  
};

//==============================================================
// Main class
//

ModSpace::ModSpace( ModSpace* parent ):
   _p( new Private ),      
   m_parent(parent),
   m_lastGCMark(0)
{}

ModSpace::~ModSpace()
{
   delete _p;
}


bool ModSpace::add( Module* mod, bool bExport, bool bOwn )
{
   TRACE( "ModSpace::add %s %s",
      mod->name().c_ize(), bExport ? "with export" :  "" 
      );
   
   Private::ModMap::iterator iter = _p->m_modmap.find( mod->name() );
   if( iter != _p->m_modmap.end() )
   {
      String name = mod->name();
      int count = 1;
      while( iter != _p->m_modmap.end() && name == iter->second->m_mod->name() )
      {
         if( mod == iter->second->m_mod )
         {
            return false;
         }

         name = mod->name() + "-";
         name.N(count);
         ++iter;
      }
      // add the module with this name.
      mod->name( name );
   }
   
   ModuleData* theData = new ModuleData( mod, bExport, bOwn );
   // add the module to the known modules...
   _p->m_modmap[mod->name()] = theData;
   // paths are unique by definition
   _p->m_modmapByUri[mod->uri()] = theData;
   
   // and to the list of modules to be served...
   _p->m_linkOrder.push_back( theData );
   _p->m_invokeOrder.push_back( theData );
   // tell the module we're in charge.
   mod->moduleSpace(this);
   
   return true;
}


void ModSpace::resolve( ModLoader* ml, Module* mod, bool bExport, bool bOwn )
{
    TRACE( "ModSpace::resolve %s %s",
      mod->name().c_ize(), bExport ? "with export" :  "" 
      );
    
   // save the module as requested.
   add( mod, bExport, bOwn );

   resolveDeps( ml, mod, bExport );
}


void ModSpace::resolveDeps( ModLoader* ml, Module* mod, bool bExport )
{   
    TRACE( "ModSpace::resolveDeps %s %s",
      mod->name().c_ize(), bExport ? "with export" :  "" 
      );
    
   // scan the dependencies.
   Module::Private* prv = mod->_p;   
   Module::Private::ReqMap &reqs = prv->m_reqs;
   
   Module::Private::ReqMap::iterator req_iter = reqs.begin();
   while( req_iter != reqs.end() )
   {
      Module::Private::Request* req = req_iter->second;
      // Did we already load the module?
      // Skip also generic import requests.
      if( req->m_uri != "" && req->m_module == 0 )
      {
         ModuleData* dlmod = findModuleData( req->m_uri,req->m_bIsUri );
         Module* lmod = 0;
         
         if( dlmod == 0 )
         {
            req->m_module = req->m_bIsUri ? ml->loadFile( req->m_uri ) : 
                                  ml->loadName( req->m_uri );
            // we can't find the module.
            if( req->m_module == 0 )
            {
               TRACE1( "ModSpace::resolveDeps failed to load request %s %s",
                     req->m_uri.c_ize(), req->m_bIsUri ? "(as uri)" :  "(as name)" 
                     );
               
               throw new LinkError( ErrorParam(e_mod_notfound, 0 )
                  .module( mod->uri() )
                  .extra(req->m_uri) );
            }
            else
            {
               TRACE1( "ModSpace::resolveDeps loaded request %s %s",
                     req->m_uri.c_ize(), req->m_bIsUri ? "(as uri)" :  "(as name)" 
                     );
               resolve( ml, req->m_module, req->m_loadMode == e_lm_load, true );
            }            
         }
         else
         {
            TRACE1( "ModSpace::resolveDeps already loaded module %s %s",
                  req->m_uri.c_ize(), req->m_bIsUri ? "(as uri)" :  "(as name)" 
                  );
            req->m_module = dlmod->m_mod;
            // should we promote the module? 
            if( req->m_loadMode == e_lm_load )
            {
               TRACE1( "ModSpace::subResolve promoted module %s", req->m_uri.c_ize() );
               dlmod->m_bExport = true;
            }
         }
       
         // lmod cant' be zero -- we should have thrown in that case.
         fassert( lmod != 0 );         
      }
      ++req_iter;
   }
}


Error* ModSpace::link()
{
//   TRACE( "ModSpace::link start" );
   
   Error* link_errors = 0;
   
   // first, publish all the exports.
   Private::ModList& list = _p->m_linkOrder;
   Private::ModList::iterator iter = list.begin();
   while( iter != list.end() )
   {
      ModuleData* md = *iter;
      Module* mod = md->m_mod;
      
      // shall we export something?
      if( md->m_bExport && ! md->m_bExported )
      {
         md->m_bExported  = true;
         exportFromModule( mod, link_errors );
      }
      
      ++iter;
   }

   // then resolve the imports.
   iter = list.begin();
   while( iter != list.end() )
   {
      Module* mod = (*iter)->m_mod;
      linkImports( mod, link_errors );
      ++iter;
   }
   
   // then clear the list
   list.clear();
   
   // in case of error, free the run order that was generated.
   if ( link_errors != 0 )
   {
      _p->m_invokeOrder.clear();
   }
   // and return
   return link_errors;
}


void ModSpace::exportFromModule( Module* mod, Error*& link_errors )
{
   TRACE1( "ModSpace::exportFromModule %s", mod->name().c_ize() );
   Module::Private::GlobalsMap::const_iterator iter, iterEnd;


   if( mod->exportAll() )
   {
      const Module::Private::GlobalsMap& syms = mod->_p->m_gSyms;         
      iter = syms.begin();
      iterEnd = syms.end();
   }
   else
   {
      const Module::Private::GlobalsMap& syms = mod->_p->m_gExports;
      iter = syms.begin();
      iterEnd = syms.end();
   }

   while( iter != iterEnd )
   {
      Symbol* sym = iter->second;
      // ignore "private" symbols
      if( ! (sym->name().startsWith("_") || sym->type() != Symbol::e_st_global) )
      {
         Error* e = exportSymbol( mod, sym );
         if( e != 0 )
         {
            addLinkError( link_errors, e );
         }
      }
      ++iter;   
   }
}


Error* ModSpace::exportSymbol( Module* mod, Symbol* sym )
{
   Private::ExportSymMap::iterator iter = _p->m_symMap.find( sym->name() );
   if( iter != _p->m_symMap.end() )
   {
      // name clash!
      return new LinkError( ErrorParam( e_already_def )
         .origin(ErrorParam::e_orig_linker)
         .module( mod->uri() )
         .line( sym->declaredAt() )
         .symbol(sym->name())
         .extra( "in " + iter->second.m_mod->name() ));
   }
   
   // link the symbol
   uint32 id = _p->m_values.length();
   TRACE1( "ModSpace::exportSymbol exporting %s.%s with id %d", 
         mod->name().c_ize(), sym->name().c_ize(), id );
   
   _p->m_symMap[ sym->name() ] = Private::ExportSymEntry(mod,sym);
   _p->m_values.append( *sym->defaultValue() );
   
   sym->define( Symbol::e_st_global, id );
   sym->defaultValue( &_p->m_values.at(id) );
   
   return 0;
}


void ModSpace::gcMark( uint32  )
{
   //TODO
}

void ModSpace::linkImports(Module* mod, Error*& link_errors)
{
   TRACE1( "ModSpace::linkImports importing requests of %s", mod->name().c_ize());
   
   Module::Private::ReqMap &reqs = mod->_p->m_reqs;   
   Module::Private::ReqMap::iterator req_iter = reqs.begin();
   while( req_iter != reqs.end() )
   {
      Module::Private::Request::DepMap& deps = req_iter->second->m_deps;
      
      // generic imports?
      if( req_iter->first == "" )
      {
         Module::Private::Request::DepMap::iterator dep_iter = deps.begin();
         while( dep_iter != deps.end() )
         {
            Module::Private::Dependency* dep = dep_iter->second;
            const String& symName = dep->m_remoteName;
            Module* declarer;
            Symbol* sym = findExportedSymbol( symName, declarer );
            if( sym != 0 )
            {
               dep->resolved( declarer, sym );
               // do we have some errors?
               Module::Private::Dependency::ErrorList::iterator ierr = dep->m_errors.begin();
               while( ierr != dep->m_errors.end() ) {
                  addLinkError(link_errors, *ierr);
                  ++ierr;
               }
               dep->clearErrors();
            }
            else
            {
               LinkError* le = new LinkError( ErrorParam( e_undef_sym )
                  .origin(ErrorParam::e_orig_linker)
                  .module( mod->uri() )
                  .line( dep->m_symbol->declaredAt() )
                  .symbol(sym->name())
                  .extra( "in " + mod->name() ));
               addLinkError( link_errors, le );
            }
            
            ++dep_iter;
         }
      }
   }
}


Symbol* ModSpace::findExportedSymbol( const String& symName, Module*& declarer )
{
   Private::ExportSymMap::iterator iter = _p->m_symMap.find( symName );
   
   if ( iter != _p->m_symMap.end() )
   {
      Private::ExportSymEntry& entry = iter->second;
      declarer = entry.m_mod;
      return entry.m_sym;
   }
   
   if( m_parent != 0 )
   {
      return m_parent->findExportedSymbol( symName, declarer );
   }
   
   return 0;
}


bool ModSpace::readyVM( VMContext* ctx )
{
   bool someRun = false;
   Private::ModList& mods = _p->m_invokeOrder;
   
   // insertion goes first to last because execution goes last to first.   
   Private::ModList::const_iterator imods = mods.begin();
   while( imods != mods.end() )
   {
      const Module* mod = (*imods)->m_mod;
      // TODO -- invoke the init methods.
      
      // TODO -- specific space for Main.
      Function* main = mod->getFunction("__main__");
      if( main != 0 )
      {
         ctx->call( main, 0 );
         someRun = true;
      }
            
      ++imods;
   }
   
   return someRun;
}

//=============================================================
//
//

Module* ModSpace::findByName( const String& name, bool& bExport ) const
{
   ModuleData* md = findModuleData( name, false );
   if( md != 0 )
   {
      bExport = md->m_bExport;
      return md->m_mod;
   }
   
   return 0;
}
      
   
Module* ModSpace::findByURI( const String& uri, bool& bExport ) const
{
   ModuleData* md = findModuleData( uri, true );
   if( md != 0 )
   {
      bExport = md->m_bExport;
      return md->m_mod;
   }
   
   return 0;
}


ModSpace::ModuleData* ModSpace::findModuleData( const String& name, bool isUri ) const
{
   const Private::ModMap* mm = isUri ? &_p->m_modmapByUri : &_p->m_modmap;
   
   Private::ModMap::const_iterator iter = mm->find( name );
   if( iter == mm->end() )
   {
      if( m_parent != 0 )
      {
         return m_parent->findModuleData( name, isUri );
      }
      
      return 0;
   }
   
   return iter->second;
}

void ModSpace::addLinkError( Error*& top, Error* newError )
{
   if ( top == 0 ) 
   {
      top = new GenericError( ErrorParam( e_link_error, __LINE__, SRC ) );
   }

   top->appendSubError(newError);
}  

}

/* end of modspace.cpp */

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

#define SRC "engine/modspace.cpp"

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
#include <falcon/importdef.h>
#include <falcon/pseudofunc.h>

#include <falcon/errors/unserializableerror.h>

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
{
   m_loader = new ModLoader(this);
}


ModSpace::~ModSpace()
{
   delete _p;
   delete m_loader;
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


void ModSpace::resolve( Module* mod, bool bExport, bool bOwn )
{
    TRACE( "ModSpace::resolve %s %s",
      mod->name().c_ize(), bExport ? "with export" :  "" 
      );
    
   // save the module as requested.
   add( mod, bExport, bOwn );

   resolveDeps( mod );
}


void ModSpace::resolveImportDef( ImportDef* def, Module* requester ) 
{
   TRACE( "ModSpace::resolveImportDef for %s", def->sourceModule().c_ize() );
   
   Module* sourcemod = 0;
   
   // do we have a module to load?
   if( def->sourceModule().size() != 0 )
   {
      ModuleData* sourcedt = findModuleData( def->sourceModule(), def->isUri() );
      
      if ( sourcedt == 0 )
      {
         sourcemod = def->isUri() ? m_loader->loadFile( def->sourceModule() ) : 
                                  m_loader->loadName( def->sourceModule() );
         
         if( sourcemod == 0 )
         {
            TRACE1( "ModSpace::resolveImportDef failed to load request %s %s",
                  def->sourceModule().c_ize(),
                  def->isUri() ? "(as uri)" :  "(as name)" 
                  );

            Error* em = new LinkError( ErrorParam(e_mod_notfound, 0 )
               .origin( ErrorParam::e_orig_linker )
               .extra( def->sourceModule() ) );            
            if( requester ) em->module( requester->uri() );
            
            throw em;
         }
         
         // be sure the loaded module deps are resolved as well.
         resolve( sourcemod, def->isLoad(), true );
      }
      else {
         sourcemod = sourcedt->m_mod;
      }
      
      if ( requester != 0 )
      {
         ModRequest* req = requester->_p->m_mrmap[def->sourceModule()];
         if( req != 0 )
         {
            req->m_module = sourcemod;
         }
      }
   }   
}


void ModSpace::resolveDeps( Module* mod)
{   
   TRACE( "ModSpace::resolveDeps %s", mod->name().c_ize() );
    
   // scan the dependencies.
   Module::Private* prv = mod->_p;
   Module::Private::ReqList &reqs = prv->m_mrlist;
   
   Module::Private::ReqList::iterator ri = reqs.begin();
   while( ri != reqs.end() )
   {
      ModRequest* req = *ri;
      
      // is this a request to load or import a module?
      if ( req->m_module == 0 )
      {
         // have we got the module here?
         ModuleData* md = findModuleData( req->m_name, req->m_bIsURI );
         
         // No -- try to load.
         if( md == 0 )
         {
            req->m_module = req->m_bIsURI ? 
                                 m_loader->loadName( req->m_name ) : 
                                 m_loader->loadFile( req->m_name );

            if( req->m_module == 0 )
            {
               TRACE1( "ModSpace::resolveDeps failed to load request %s %s",
                     req->m_name.c_ize(),
                     req->m_bIsURI ? "(as uri)" : "(as name)" 
                     );

               Error* em = new LinkError( ErrorParam(e_mod_notfound, 0, mod->uri() )
                  .origin( ErrorParam::e_orig_linker )
                  .extra( req->m_name ) );            

               throw em;
            }

            // be sure the loaded module deps are resolved as well.
            resolve( req->m_module, req->m_isLoad, true );
         }
         else 
         {
            req->m_module = md->m_mod;
            
            // eventually promote to load.
            if( req->m_isLoad && ! md->m_bExport )
            {
               md->m_bExport = true;
               _p->m_linkOrder.push_back( md );
            }
         }
      }     
      
      ++ri;
   }
}


Error* ModSpace::link()
{
   MESSAGE( "ModSpace::link start" );
   
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
      linkNSImports( mod );
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


Error* ModSpace::linkModuleImports( Module* mod )
{
   TRACE( "ModSpace::linkModuleImports start on %s", mod->uri().c_ize() );
   Error* link_errors = 0;
   linkImports( mod, link_errors );
   
   if( link_errors == 0 )
   {
      linkNSImports( mod );
   }
   
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
   TRACE1( "ModSpace::exportSymbol exporting %s.%s", 
         mod->name().c_ize(), sym->name().c_ize() );
   
   // if the symbol is an extern symbol, then we have to get its
   // reference.
   
   _p->m_symMap[ sym->name() ] = Private::ExportSymEntry(mod, sym);
   return 0;
}


void ModSpace::gcMark( uint32  )
{
   //TODO
}


void ModSpace::linkImports(Module* mod, Error*& link_errors)
{
   TRACE1( "ModSpace::linkImports importing requests of %s", mod->name().c_ize());
   
   // scan the dependencies.
   Module::Private* prv = mod->_p;   
   Module::Private::DepList &deps = prv->m_deplist;
   Module::Private::DepList::const_iterator dep_iter = deps.begin();
   while( dep_iter != deps.end() )
   {
      Module::Private::Dependency* dep = *dep_iter;
      // ignore already resolved symbols (?)
      if( dep->m_resSymbol == 0 )
      {
         // now, we have some symbol to import here. Are they general or specific?
         if( dep->m_idef != 0 )
         {
            linkSpecificDep( mod, dep, link_errors );
         }
         else
         {
            linkGeneralDep( mod, dep, link_errors );
         }
      }
      ++dep_iter;
   }
}


void ModSpace::linkSpecificDep( Module* asker, void* def, Error*& link_errors )
{
   Module::Private::Dependency* dep = (Module::Private::Dependency*) def;
   
   Symbol* sym = 0;
   Module* sourcemod = 0;
      
   fassert( dep->m_idef != 0 );
   if( dep->m_idef->modReq() != 0 )
   {
      fassert( dep->m_idef->modReq()->module() != 0 );

      sourcemod = dep->m_idef->modReq()->module();

      // in case we have the module, search the symbol there. 
      sym = sourcemod->getGlobal( dep->m_sourceName );            
   }
   else {
      sym = findExportedSymbol( dep->m_sourceName, sourcemod );
   }
   
   // not found? we have a link error.
   if( sym == 0 )
   {
      Error* em = new LinkError( ErrorParam(e_undef_sym, 0, asker->uri() )
         .origin( ErrorParam::e_orig_linker )
         .symbol( dep->m_sourceName )
         .extra( "in " + dep->m_sourceName ) );

      addLinkError( link_errors, em );
      return;
   }
   
   // Ok, we have the symbol. Now we must tell the requester we have found it
   Error* em = dep->onResolved( asker, sourcemod, sym );
   if( em != 0 )
   {
      addLinkError( link_errors, em );
      return;
   }
   
   // link the value -- if needed
   if ( dep->m_symbol != 0 )
   {
      Variable* variable = sym->getVariable(0);
      fassert( variable != 0 );
      dep->m_symbol->resolved( variable );
   }

}


void ModSpace::linkGeneralDep(Module* asker, void* def, Error*& link_errors)
{   
   Module::Private::Dependency* dep = (Module::Private::Dependency*) def;
   
   // in case we have the module, search the symbol there.     
   Module* declarer = 0;
   const String& symName = dep->m_sourceName;
   Symbol* sym;

   sym = findExportedOrGeneralSymbol( asker, symName, declarer );
   // not found? we have a link error.
   if( sym == 0 )
   {
      Error* em = new LinkError( ErrorParam(e_undef_sym, 0, asker->uri() )
         .origin( ErrorParam::e_orig_linker )
         .symbol( symName ) );

      addLinkError( link_errors, em );
      return;
   }

   // Ok, we have the symbol. Now we must tell the requester we have found it
   Error* em = dep->onResolved( asker, declarer, sym );
   if( em != 0 )
   {
      addLinkError( link_errors, em );
   }
}



void ModSpace::linkNSImports(Module* mod )
{
   TRACE1( "ModSpace::linkNSImports honoring namespace imports requests of %s", 
         mod->name().c_ize());
      
   // scan the dependencies.
   Module::Private* prv = mod->_p;   
   
   Module::Private::NSImportList::iterator nsi = prv->m_nsimports.begin();
   while( nsi != prv->m_nsimports.end() )
   {
      Module::Private::NSImport* ns = *nsi;
      
      if( ! ns->m_bPerformed )
      {
         ns->m_bPerformed = true;
         fassert( ns->m_def != 0);
         fassert( ns->m_def->modReq() != 0 );
         fassert( ns->m_def->modReq()->module() != 0 );
         Module* srcMod = ns->m_def->modReq()->module();
                  
         String srcName, tgName;
         if( ns->m_from.size() != 0 )
         {
            srcName = ns->m_from + ".";
         }
         
         Module::Private::GlobalsMap& srcGlobals = srcMod->_p->m_gSyms;         
         Module::Private::GlobalsMap::iterator glb = srcGlobals.lower_bound( srcName );
         while( glb != srcGlobals.end() )
         {
            Symbol* sym = glb->second;
            if( ! sym->name().startsWith( srcName ) )
            {
               // we're done
               break;
            }
            
            // find the target name.
            tgName = ns->m_to.size() == 0 ? sym->name() : ns->m_to + "." + sym->name();
            // import it.
            Module::Private::GlobalsMap::iterator myglb = mod->_p->m_gSyms.find( tgName );
            if( myglb == mod->_p->m_gSyms.end() )
            {
               // new symbol, make it external -- declared in main function of the module
               Symbol* esym = new Symbol( tgName, Symbol::e_st_extern );
               mod->_p->m_gSyms[tgName] = esym;
               Variable* var = sym->getVariable(0);
               fassert( var != 0 );
               esym->resolved( var );
            }
            else
            {
               // just link it 
               Symbol* esym = myglb->second;
               Variable* var = sym->getVariable(0);
               fassert( var != 0 );
               esym->resolved( var );
               
               // eventually, make it resolved in dependencies, 
               // -- so we don't search for it elsewhere.
               Module::Private::DepMap::iterator di = mod->_p->m_depsBySymbol.find( tgName );
               if( di != mod->_p->m_depsBySymbol.end() )
               {
                  Module::Private::Dependency* dep = di->second;
                  if( dep->m_resSymbol == 0 )
                  {
                     dep->m_resSymbol = esym;
                     dep->onResolved( mod, srcMod, esym );
                  }
               }
            }
            
            ++glb;
         }
      }
      
      ++nsi;
   }
}


Symbol* ModSpace::findExportedOrGeneralSymbol( Module* asker, const String& symName, Module*& declarer )
{  
   Symbol* sym = findExportedSymbol( symName, declarer );      
   
   if( sym == 0 )
   {
      Module::Private::ReqList::iterator geni = asker->_p->m_genericMods.begin();
      // see if we have a namespace.
      length_t dotpos = symName.rfind( '.' );
      String nspace, nsname;
      
      if( dotpos != String::npos )
      {
         nspace = symName.subString(0,dotpos);
         nsname = symName.subString(dotpos+1);
      }

      while( geni != asker->_p->m_genericMods.end() )
      {
         ModRequest* req = *geni;
         
         if ( req->m_module != 0 )
         {
            if( dotpos == String::npos )
            {
               sym = req->m_module->getGlobal( symName );
               if( sym != 0 )
               {
                  declarer = req->m_module;
                  break;
               }
            }
            else
            {
               // search the symbol in a namespace.
               for( int idi = 0; idi < req->importDefCount(); ++ idi )
               {
                  ImportDef* def =req->importDefAt( idi );
                     
                  if ( def->isNameSpace() && def->target() == nspace )
                  {
                     sym = req->m_module->getGlobal( nsname );
                     if( sym != 0 )
                     {
                        declarer = req->m_module;
                        break;
                     }
                  }
               }
            }
            
         }
         ++geni;
      }
   }
   
   return sym;
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
      
      if( ! mod->isMain() )
      {
         Function* main = mod->getFunction("__main__");
         if( main != 0 )
         {
            ctx->call( main, 0 );
            someRun = true;
         }
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


Module* ModSpace::retreiveDynamicModule( 
      const String& moduleUri, 
      const String& moduleName,  
      bool &addedMod )
{
   Module* clsContainer = 0;
   // priority in search and load are inverted:
   // search by name -- by uri / load by uri -- by name.


   // first, try to search by name
   if( moduleName != "" )
   {
      clsContainer = this->findByName( moduleName );
      
      // if not found, see if we can find by uri
      if( clsContainer == 0 && moduleUri != "" )
      {
         // is the URI around?
         clsContainer = this->findByURI( moduleUri );
         if( clsContainer == 0 )
         {
            // then try to load the module.
            clsContainer = m_loader->loadFile( moduleUri, ModLoader::e_mt_none, true );
            // to be linked?
            addedMod = clsContainer != 0;
         }
      }
      
      // Not loaded? -- try to load by name.
      if( clsContainer == 0 )
      {
         // then try to load the module.
         clsContainer = m_loader->loadName( moduleName );
         // to be linked?
         addedMod = clsContainer != 0;
      }
   }

   return clsContainer;
}


Mantra* ModSpace::findDynamicMantra( 
      const String& moduleUri, 
      const String& moduleName, 
      const String& className, 
      bool &addedMod )
{
   static Engine* eng = Engine::instance();
   
   Module* clsContainer = retreiveDynamicModule( moduleUri, moduleName, addedMod );
   Mantra* theClass;

   // still no luck?
   if( clsContainer == 0 )
   {
      if( moduleName != "" )
      {
         // we should have found a module -- and if the module has a URI,
         // then it has a name.
         throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( "Can't find module " + 
                  moduleName + " for " + className ));
      }

      Symbol* sym = this->findExportedSymbol( className );
      if( sym == 0 )
      {
         theClass = eng->getMantra( className );
         if( theClass == 0 )
         {
            throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( "Unavailable mantra " + className ));
         }
      }
      else 
      {
         if( sym->defaultValue().isClass() || sym->defaultValue().isFunction() )
         {
            theClass = static_cast<Mantra*>(sym->defaultValue().asInst());
         }
         else
         {
             throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( "Symbol is not mantra " + className ));
         }
      }
   }
   else
   {
      // we are sure about the sourceship of the class...
      theClass = clsContainer->getMantra( className );
      if( theClass == 0 )
      {
         String expl = "Unavailable mantra " + 
                  className + " in " + clsContainer->uri();
         delete clsContainer; // the module is not going anywhere.
         throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( expl ));
      }

      // shall we link a new module?
      if( addedMod )
      {
         this->resolve( clsContainer, false, true );
      }
   }
   
   return theClass;
}

}

/* end of modspace.cpp */

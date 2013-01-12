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
#include <falcon/process.h>
#include <falcon/vm.h>
#include <falcon/stdsteps.h>
#include <falcon/synfunc.h>
#include <falcon/psteps/stmtreturn.h>

#include <falcon/errors/unserializableerror.h>
#include <falcon/errors/ioerror.h>

#include <map>
#include <deque>

#include "module_private.h"

namespace Falcon {
   

class ModSpace::Private
{
public:
   class ExportSymEntry{
   public:
      Module* m_mod;
      Item* m_value;
      
      ExportSymEntry(): m_mod(0), m_value(0) {}
      
      ExportSymEntry(Module* mod, Item* item ):
         m_mod(mod),
         m_value(item)
      {}
      
      ExportSymEntry( const ExportSymEntry& other ):
         m_mod(other.m_mod),
         m_value(other.m_value)
      {}
   };
   
   typedef std::map<String, ExportSymEntry> ExportSymMap;
   ExportSymMap m_symMap;
   
   typedef std::map<String, Module*> ModMap;
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
         iter->second->decref();
         ++iter;
      }
   }  
};

//==============================================================
// Main class
//

ModSpace::ModSpace( VMachine* owner, ModSpace* parent ):
   _p( new Private ),   
   m_parent(parent),
   m_lastGCMark(0),
   m_stepLoader(this),
   m_stepResolver(this),
   m_stepDynModule(this),
   m_stepDynMantra(this),
   m_stepExecMain(this)
{
   m_vm = owner;
   m_loader = new ModLoader(this);
   SynFunc* sf = new SynFunc("$loadModule");
   sf->syntree().append( new StmtReturn );
   m_loaderFunc = sf;
}


ModSpace::~ModSpace()
{
   delete _p;
   delete m_loader;
   delete m_loaderFunc;
}


Process* ModSpace::loadModule( const String& name, bool isUri,  bool isMain )
{

   Process* process = m_vm->createProcess();
   VMContext* tgtContext = process->mainContext();
   tgtContext->callInternal( m_loaderFunc, 0 );
   tgtContext->pushCode(&Engine::instance()->stdSteps()->m_returnFrameWithTop);
   loadSubModule( name, isUri, isMain, tgtContext );

   return process;
}


void ModSpace::loadSubModule( const String& name, bool isUri, bool isMain, VMContext* tgtContext )
{
   tgtContext->pushCode( &m_stepLoader );
   if( ! isMain ) {
      // skip the main module set step
      tgtContext->currentCode().m_seqId = 1;
   }

   bool loading;

   if( isUri )
   {
      loading = m_loader->loadFile( tgtContext, name, ModLoader::e_mt_none, true );
   }
   else
   {
      loading = m_loader->loadName( tgtContext, name );
   }

   if( ! loading ) {
      tgtContext->returnFrame();
      throw new IOError( ErrorParam( e_mod_notfound, __LINE__, SRC )
               .extra( name ) );
   }
}


void ModSpace::add( Module* mod )
{
   TRACE( "ModSpace::add %s", mod->name().c_ize() );

   store( mod );

   Error* le = 0;
   exportFromModule(mod, le );
   if( le != 0 ) {
      throw le;
   }
}


void ModSpace::store( Module* mod )
{
   TRACE( "ModSpace::store %s", mod->name().c_ize() );
   mod->incref();

   Private::ModMap::iterator iter = _p->m_modmap.find( mod->name() );
   if( iter != _p->m_modmap.end() )
   {
      String name = mod->name();
      int count = 1;
      while( iter != _p->m_modmap.end() && name == iter->second->name() )
      {
         if( mod == iter->second )
         {
            // we already have this module.
            mod->decref();
            return;
         }

         name = mod->name() + "-";
         name.N(count);
         ++iter;
      }
      // add the module with this name.
      mod->name( name );
   }
   
   // add the module to the known modules...
   _p->m_modmap[mod->name()] = mod;
   // paths are unique by definition
   // -- i.e. we don't care if we have multiple modules with the same path
   _p->m_modmapByUri[mod->uri()] = mod;
   
   // tell the module we're in charge.
   mod->modSpace(this);
}


void ModSpace::resolveDeps( VMContext* ctx, Module* mod )
{
   static StdSteps* steps = Engine::instance()->stdSteps();
   static Class* modcls = Engine::instance()->moduleClass();

   ctx->pushData( Item( modcls, mod ) );
   ctx->pushCode( &steps->m_pop );
   ctx->pushCode( &m_stepResolver );
}


void ModSpace::exportFromModule( Module* mod, Error*& link_errors )
{
   TRACE1( "ModSpace::exportFromModule %s", mod->name().c_ize() );

   uint32 count = mod->globals().size();
   for( uint32 i = 0; i < count; ++i )
   {
      VarDataMap::VarData* vd = mod->globals().getGlobal(i);
      // ignore "private" symbols
      if( ! vd->m_name.startsWith("_") && (mod->exportAll() || vd->m_bExported))
      {
         Error* e = exportSymbol( mod, vd->m_name, vd->m_var );
         if( e != 0 )
         {
            addLinkError( link_errors, e );
         }
      }
   }
}


Error* ModSpace::exportSymbol( Module* mod, const String& name, const Variable& var )
{
   TRACE1( "ModSpace::exportSymbol %s", name.c_ize());

   Private::ExportSymMap::iterator iter = _p->m_symMap.find( name );
   if( iter != _p->m_symMap.end() )
   {
      // name clash!
      return new LinkError( ErrorParam( e_already_def )
         .origin(ErrorParam::e_orig_linker)
         .module( mod->uri() )
         .line( var.declaredAt() )
         .symbol(name )
         .extra( "in " + iter->second.m_mod->name() ));
   }
   
   // link the symbol
   TRACE1( "ModSpace::exportSymbol exporting %s.%s", 
         mod->name().c_ize(), name.c_ize() );
   
   // if the symbol is an extern symbol, then we have to get its
   // reference.
   
   _p->m_symMap[ name ] = Private::ExportSymEntry(mod, mod->getGlobalValue(var.id()));
   return 0;
}


void ModSpace::gcMark( uint32  )
{
   //TODO
}


void ModSpace::importInModule(Module* mod, Error*& link_errors)
{
   TRACE1( "ModSpace::importInModule importing requests of %s", mod->name().c_ize());
   
   // scan the dependencies.
   Module::Private* prv = mod->_p;   
   Module::Private::DepList &deps = prv->m_deplist;
   Module::Private::DepList::const_iterator dep_iter = deps.begin();
   while( dep_iter != deps.end() )
   {
      Module::Private::Dependency* dep = *dep_iter;
      // some dependency get sterilized.
      if( dep->m_id >= 0  )
      {
         // now, we have some symbol to import here. Are they general or specific?
         if( dep->m_idef != 0 )
         {
            importSpecificDep( mod, dep, link_errors );
         }
         else
         {
            importGeneralDep( mod, dep, link_errors );
         }
      }
      ++dep_iter;
   }
}


void ModSpace::importSpecificDep( Module* asker, void* def, Error*& link_errors )
{
   Module::Private::Dependency* dep = (Module::Private::Dependency*) def;
   
   TRACE( "ModSpace::importSpecificDep %s", dep->m_sourceName.c_ize() );

   Item* value = 0;
   Module* sourcemod = 0;
      
   fassert( dep->m_idef != 0 );
   if( dep->m_idef->modReq() != 0 )
   {
      fassert( dep->m_idef->modReq()->module() != 0 );

      sourcemod = dep->m_idef->modReq()->module();

      // in case we have the module, search the symbol there. 
      value = sourcemod->getGlobalValue( dep->m_sourceName );
   }
   else {
      value = findExportedValue( dep->m_sourceName, sourcemod );
   }
   
   // not found? we have a link error.
   if( value == 0 )
   {
      Error* em = new LinkError( ErrorParam(e_undef_sym, 0, asker->uri() )
         .origin( ErrorParam::e_orig_linker )
         .symbol( dep->m_sourceName )
         .extra( "in " + dep->m_sourceName ) );

      addLinkError( link_errors, em );
      return;
   }
   
   // Ok, we have the symbol. Now we must tell the requester we have found it
   try
   {
      asker->resolveExternValue( dep->m_sourceName, sourcemod, value );
   }
   catch( Error* em )
   {
      addLinkError( link_errors, em );
      em->decref();
   }
}


void ModSpace::importGeneralDep(Module* asker, void* def, Error*& link_errors)
{
   Module::Private::Dependency* dep = (Module::Private::Dependency*) def;
   
   // in case we have the module, search the symbol there.     
   Module* declarer = 0;
   const String& symName = dep->m_sourceName;
   Item* value;

   TRACE( "ModSpace::importGeneralDep %s", symName.c_ize() );

   value = findExportedOrGeneralValue( asker, symName, declarer );

   // not found? we have a link error.
   if( value == 0 )
   {
      Error* em = new LinkError( ErrorParam(e_undef_sym, 0, asker->uri() )
         .origin( ErrorParam::e_orig_linker )
         .symbol( symName ) );

      addLinkError( link_errors, em );
      return;
   }

   // Ok, we have the symbol. Now we must tell the requester we have found it
   try {
      asker->resolveExternValue( symName, declarer, value );
   }
   catch( Error* em )
   {
      addLinkError( link_errors, em );
      em->decref();
   }
}



void ModSpace::importInNS(Module* mod )
{
   TRACE1( "ModSpace::importInNS honoring namespace imports requests of %s",
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
         
         srcMod->exportNS( ns->m_from, mod, ns->m_to );
      }
      
      ++nsi;
   }
}


Item* ModSpace::findExportedOrGeneralValue( Module* asker, const String& symName, Module*& declarer )
{  
   Item* sym = findExportedValue( symName, declarer );
   
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
               sym = req->m_module->getGlobalValue( symName );
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
                     sym = req->m_module->getGlobalValue( nsname );
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

   
Item* ModSpace::findExportedValue( const String& symName, Module*& declarer )
{
   Private::ExportSymMap::iterator iter = _p->m_symMap.find( symName );
   
   if ( iter != _p->m_symMap.end() )
   {
      Private::ExportSymEntry& entry = iter->second;
      declarer = entry.m_mod;
      return entry.m_value;
   }
      
   if( m_parent != 0 )
   {
      return m_parent->findExportedValue( symName, declarer );
   }
   
   return 0;
}


//=============================================================
//
//

Module* ModSpace::findModule( const String& name, bool isUri ) const
{
   const Private::ModMap* mm = isUri ? &_p->m_modmapByUri : &_p->m_modmap;
   
   Private::ModMap::const_iterator iter = mm->find( name );
   if( iter == mm->end() )
   {
      if( m_parent != 0 )
      {
         return m_parent->findModule( name, isUri );
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


void ModSpace::retreiveDynamicModule(
      VMContext* ctx,
      const String& moduleUri, 
      const String& moduleName )
{
   static Class* clsMod = Engine::instance()->moduleClass();

   TRACE( "ModSpace::retreiveDynamicModule %s, %s",
            moduleUri.c_ize(), moduleName.c_ize() );

   Module* clsContainer = 0;
   // priority in search and load are inverted:
   // search by name -- by uri / load by uri -- by name.


   // first, try to search by name
   if( moduleName != "" )
   {
      clsContainer = this->findByName( moduleName );
   }

   // if not found, see if we can find by uri
   if( clsContainer == 0 && moduleUri != "" )
   {
      // is the URI around?
      clsContainer = this->findByURI( moduleUri );
   }

   // still not found?
   if( clsContainer == 0 )
   {
      // then try to load the module via URI
      ctx->pushData( Item( moduleName.handler(), const_cast<String*>(&moduleName)) );
      //... delayed check
      ctx->pushCode( &m_stepDynModule );

      m_loader->loadFile( ctx, moduleUri, ModLoader::e_mt_none, true );
   }
   else
   {
      TRACE( "ModSpace::retreiveDynamicModule found module %s: %s",
                 clsContainer->name().c_ize(), clsContainer->uri().c_ize() );
      ctx->pushData( Item( clsMod, clsContainer ) );
   }
}


void ModSpace::findDynamicMantra(
      VMContext* ctx,
      const String& moduleUri, 
      const String& moduleName, 
      const String& className )
{
   static Engine* eng = Engine::instance();

   TRACE( "ModSpace::findDynamicMantra %s, %s, %s",
            moduleUri.c_ize(), moduleName.c_ize(), className.c_ize() );

   if( moduleUri.size() == 0 && moduleName.size() == 0 )
   {
      Mantra* result = eng->getMantra(className);
      if( result != 0 )
      {
         MESSAGE( "Found required mantra in engine." );
         ctx->pushData( Item( result->handler(), result ) );
         return;
      }
      else {
         // we can't find this stuff in modules -- as we don't have a module to search for.
         ctx->pushData( Item() );
         return;
      }
   }

   // push the mantra resolver step.
   ctx->pushData( Item( className.handler(), const_cast<String*>(&className)) );
   ctx->pushCode( &m_stepDynMantra );

   // and then start resolving the module
   retreiveDynamicModule( ctx, moduleUri, moduleName );
}

//================================================================================
// Psteps
//================================================================================


void ModSpace::PStepLoader::apply_( const PStep* self, VMContext* ctx )
{
   Error* error = 0;
   const ModSpace::PStepLoader* pstep = static_cast<const ModSpace::PStepLoader* >(self);

   int& seqId = ctx->currentCode().m_seqId;
   // the module we're working on is at top data stack.
   Module* mod = static_cast<Module*>(ctx->topData().asInst());
   ModSpace* ms = pstep->m_owner;

   switch( seqId )
   {
   // first step: set flag for execution of the main function.
   case 0:
      TRACE( "ModSpace::PStepLoader::apply_ step 0 on module %s", mod->name().c_ize() );
      seqId++;
      // non-main modules start at step 1
      mod->setMain( true );
      //fallthrough

      // second step: add module to the module space.
   case 1:
      TRACE( "ModSpace::PStepLoader::apply_ step 1 on module %s", mod->name().c_ize() );
      seqId++;
      // store the module
      ms->store( mod );

      // perform exports, so imports can find them on need.
      if( mod->usingLoad() ) {
         ms->exportFromModule( mod, error );

         if( error != 0 ) {
            throw error;
         }
      }

      //do we have some dependency?
      if( mod->depsCount() > 0 )
      {
         // push the code that will solve it.
         ctx->pushCode( &ms->m_stepResolver );
         return;
      }

      // fall-through; third step, link
   case 2:
      TRACE( "ModSpace::PStepLoader::apply_ step 2 on module %s", mod->name().c_ize() );
      seqId++;
      // Now that deps are resolved, do imports.
      ms->importInModule( mod, error );
      if( error != 0 ) {
         throw error;
      }
      ms->importInNS( mod );

      //TODO push init.

      // Push the main funciton only if this is not a main module.
      if( mod->getMainFunction() != 0 && ! mod->isMain() ) {
         ctx->callInternal( mod->getMainFunction(), 0 );
         return;
      }
   }

   // Third step -- we can remove self and the module.
   TRACE( "ModSpace::PStepLoader::apply_ step 2 on module %s", mod->name().c_ize() );
   // leave the module in the stack if it's main.
   if( ! mod->isMain() ) {
      ctx->popData();
   }

   ctx->popCode();
}


void ModSpace::PStepResolver::apply_( const PStep* self, VMContext* ctx )
{
   const ModSpace::PStepResolver* pstep = static_cast<const ModSpace::PStepResolver* >(self);
   int& seqId = ctx->currentCode().m_seqId;

   // the module we're working on is at top data stack.
   Module* mod = static_cast<Module*>(ctx->topData().asInst());
   ModSpace* ms = pstep->m_owner;

   uint32 depsCount = mod->depsCount();
   TRACE( "ModSpace::PStepResolver::apply_  on module %s (%d/%d)",
            mod->name().c_ize(), seqId, depsCount );

   while( seqId < (int) depsCount )
   {
      // don't alter seqid now, we might check this dependency again.
      ImportDef* idef = mod->getDep( seqId );

      // do we have a module to import?
      if( idef->sourceModule().size() != 0 && idef->modReq()->module() == 0 )
      {
         // already available?
         Module* other = ms->findModule( idef->sourceModule(), idef->isUri() );
         if( other == 0 ) {
            // No? -- load it -- and always launch main
            ms->loadSubModule( idef->sourceModule(), idef->isUri(), false, ctx );
            // -- and wait for the process to have resolved it.
            return;
         }

         idef->modReq()->module( other );
      }

      // get ready for next loop
      ++seqId;
   }

   ctx->popCode();
}


void ModSpace::PStepDynModule::apply_( const PStep* self, VMContext* ctx )
{
   // --- STACK:
   // top-1 : module name
   // top   : module or nil
   //
   const ModSpace::PStepDynModule* pstep = static_cast<const ModSpace::PStepDynModule* >(self);
   ModSpace* ms = pstep->m_owner;
   const String& moduleName = *ctx->opcodeParam(1).asString();
   int& seqId = ctx->currentCode().m_seqId;

   TRACE( "ModSpace::PStepDynModule::apply_  %d/2", seqId );

   // Not loaded?
   if( ctx->topData().isNil() )
   {
      if( seqId == 0 )
      {
         // try with the name loading
         seqId++;
         ctx->popData();
         TRACE( "ModSpace::PStepDynModule::apply_  module not found, trying to load by name %s",
                  ctx->topData().asString()->c_ize() );
         // then try to load the module.
         ms->m_loader->loadName( ctx, moduleName );
      }
      else
      {
         TRACE( "ModSpace::PStepDynModule::apply_ module \"%s\" not found giving up",
                  moduleName.c_ize() );
         //we're done -- and we have no module.

         // we should have found a module -- and if the module has a URI,
         // then it has a name.
         throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( "Can't find module " +
                  moduleName /*+ " for " + className*/ ));
      }
   }
   else
   {
      TRACE( "ModSpace::PStepDynModule::apply_  module \"%s\" found, resolving deps",
               ctx->topData().asString()->c_ize() );
      // we're done, but we have a module -- let's use the resolver code to proceed.
      // remove the name string, saving the module down in the stack.
      ctx->opcodeParam(1) = ctx->opcodeParam(0);
      ctx->popData();
      ctx->resetCode( &ms->m_stepExecMain );
      // proceed as with a newly loaded module -- consider it main (seqId=0).
      ctx->pushCode( &ms->m_stepLoader );
   }
}


void ModSpace::PStepDynMantra::apply_( const PStep* self, VMContext* ctx )
{
   static Engine* eng = Engine::instance();
   Mantra* theClass;

   ModSpace* ms = static_cast<const ModSpace::PStepDynMantra*>(self)->m_owner;
   // get the parameters
   Module* clsContainer = ctx->topData().isNil() ? 0 : static_cast<Module*>( ctx->topData().asInst() );
   const String& className = *ctx->opcodeParam(1).asString();

   // remove the module
   ctx->popData();

   // still no luck?
   if( clsContainer == 0 )
   {
      Item* sym = ms->findExportedValue( className );
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

         if( sym->isClass() || sym->isFunction() )
         {
            theClass = static_cast<Mantra*>(sym->asInst());
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
         throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime)
            .extra( expl ));
      }
   }

   ctx->pushData( Item(theClass->handler(), theClass) );
}


void ModSpace::PStepExecMain::apply_( const PStep*, VMContext* ctx )
{
   // we're called just once
   ctx->popCode();

   Module* mod = static_cast<Module*>( ctx->topData().asInst() );
   mod->setMain(false);

   if( mod->getMainFunction() != 0 ) {
      ctx->callInternal( mod->getMainFunction(), 0 );
   }
}

}

/* end of modspace.cpp */

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
#include <falcon/falconclass.h>
#include <falcon/stdhandlers.h>
#include <falcon/classes/classmodule.h>

#include <falcon/errors/unserializableerror.h>
#include <falcon/errors/ioerror.h>

#include <map>
#include <deque>

#include "module_private.h"

namespace Falcon {
   

Class* ModSpace::handler()
{
   static Class* ms = Engine::handlers()->modSpaceClass();
   return ms;
}

class ModSpace::Private
{
public:
   class ExportSymEntry{
   public:
      Module* m_mod;
      Item* m_value;
      Item m_internal;
      
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

ModSpace::ModSpace( Process* owner, ModSpace* parent ):
   _p( new Private ),   
   m_parent(parent),
   m_lastGCMark(0),
   m_mainMod(0),
   m_stepLoader(this),
   m_stepResolver(this),
   m_stepDynModule(this),
   m_stepDynMantra(this),
   m_stepExecMain(this),
   m_startLoadStep(this)
{
   m_process = owner;
   m_loader = new ModLoader(this);
   SynFunc* sf = new SynFunc("$loadModule");
   sf->syntree().append( new StmtReturn );
   m_loaderFunc = sf;

   if( parent != 0 )
   {
      parent->incref();
   }
}


ModSpace::~ModSpace()
{
   if( m_parent != 0 )
   {
      m_parent->decref();
   }

   delete _p;
   delete m_loader;
   delete m_loaderFunc;
}


Process* ModSpace::loadModule( const String& name, bool isUri,  bool asLoad, bool isMain )
{
   Process* process = m_process->vm()->createProcess();
   loadModuleInProcess( process, name, isUri, asLoad, isMain );
   return process;
}

void ModSpace::loadModuleInProcess( Process* process, const String& name, bool isUri,  bool asLoad, bool isMain, Module* loader )
{
   static Class* clsModule = Engine::handlers()->moduleClass();

   process->adoptModSpace(this);

   VMContext* tgtContext = process->mainContext();
   tgtContext->call( m_loaderFunc );
   tgtContext->pushCode(&Engine::instance()->stdSteps()->m_returnFrameWithTop);

   if( loader != 0 )
   {
      tgtContext->pushData(Item(clsModule, loader));
   }
   else {
      tgtContext->pushData( Item() );
   }
   tgtContext->pushData( FALCON_GC_HANDLE(new String(name)) );
   tgtContext->pushCode( &m_startLoadStep );
   int32 v = (isUri ? 1 : 0) | (asLoad ? 2 : 0) | (isMain ? 4 : 0);
   tgtContext->currentCode().m_seqId = v;
}

void ModSpace::loadModuleInProcess( const String& name, bool isUri, bool asLoad, bool isMain, Module* loader )
{
   loadModuleInProcess( m_process, name, isUri, asLoad, isMain, loader );
}

void ModSpace::loadModuleInContext( const String& name, bool isUri, bool isLoad, bool isMain, VMContext* tgtContext, Module* loader, bool getResult )
{
   tgtContext->pushCode( &m_stepLoader );
   tgtContext->currentCode().m_seqId = (isLoad ? 1:0) | (isMain ? 2:0) |  (getResult? 4 : 0);

   bool loading;

   if( isUri )
   {
      loading = m_loader->loadFile( tgtContext, name, ModLoader::e_mt_none, true, loader );
   }
   else
   {
      loading = m_loader->loadName( tgtContext, name, ModLoader::e_mt_none, loader );
   }

   if( ! loading ) {
      throw new IOError( ErrorParam( e_mod_notfound, __LINE__, SRC )
               .extra( name + " in " + m_loader->getSearchPath() ));
   }
}


void ModSpace::add( Module* mod )
{
   TRACE( "ModSpace::add %s", mod->name().c_ize() );

   store( mod );

   //initialize simple singletons
   int32 icount = mod->getInitCount();
   if( icount != 0 )
   {
      // prepare all the required calls.
      for( int32 i = 0; i < icount; ++i )
      {
         Class* cls = mod->getInitClass(i);
         String instName = cls->name().subString(1);
         Item* gval = mod->getGlobalValue(instName);
         if( gval != 0 )
         {
            // Success!
            gval->assignFromLocal( Item( cls, cls->createInstance() ) );
         }
      }
   }

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
   static Class* modcls = Engine::handlers()->moduleClass();

   ctx->pushData( Item( modcls, mod ) );
   ctx->pushData( Item() );
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


void ModSpace::gcMark( uint32 mark )
{
   if( m_lastGCMark != mark )
   {
      m_lastGCMark = mark;
      Private::ExportSymMap::iterator esi = _p->m_symMap.begin();
      Private::ExportSymMap::iterator esi_end = _p->m_symMap.end();

      while( esi != esi_end ) {
         esi->second.m_value->gcMark(mark);
         ++esi;
      }

      Private::ModMap::iterator mmi = _p->m_modmap.begin();
      Private::ModMap::iterator mmi_end = _p->m_modmap.end();

      while( mmi != mmi_end ) {
         mmi->second->gcMark(mark);
         ++mmi;
      }
   }
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
      Error* em = new LinkError( ErrorParam(e_undef_sym, __LINE__, SRC )
         .origin( ErrorParam::e_orig_linker )
         .module(asker->uri())
         .line( dep->m_idef->sr().line() )
         .extra( dep->m_sourceName ) );

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
      Error* em = new LinkError( ErrorParam(e_undef_sym, __LINE__, SRC )
         .origin( ErrorParam::e_orig_linker )
         .module(asker->uri())
         .line( dep->m_defLine )
         .extra( symName ) );

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
   Item* item = findExportedValue( symName, declarer );
   
   if( item == 0 )
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
               item = req->m_module->getGlobalValue( symName );
               if( item != 0 )
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
                     item = req->m_module->getGlobalValue( nsname );
                     if( item != 0 )
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

   if( item == 0 ) {
      Mantra* m = Engine::instance()->getMantra(symName);
      if( m != 0 ) {
         //lock
         Private::ExportSymMap::iterator entry = _p->m_symMap.insert(std::make_pair(symName, Private::ExportSymEntry() ) ).first;
         item = entry->second.m_value = &entry->second.m_internal;
         item->setUser(m->handler(), m);
      }

   }
   return item;
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
   static Class* clsMod = Engine::handlers()->moduleClass();

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

   void ModSpace::enumerateIStrings( IStringEnumerator& cb ) const
   {
      Private::ModMap::const_iterator miter = _p->m_modmap.begin();
      while( miter != _p->m_modmap.end() )
      {
         miter->second->enumerateIStrings(cb);
         ++miter;
      }

      if( m_parent != 0 )
      {
         m_parent->enumerateIStrings(cb);
      }
   }

//================================================================================
// Psteps
//================================================================================

static void pushAttribs( VMContext* ctx, const AttributeMap& map, bool& bDone )
{
   static PStep* attribStep = &Engine::instance()->stdSteps()->m_fillAttribute;

   uint32 count = map.size();
   for( uint32 i = 0; i < count; ++i )
   {
      Attribute* attrib = map.get( i );
      if( attrib->generator() != 0 ) {
         bDone = true;
         ctx->pushData( Item(Attribute::CLASS_NAME, attrib ) );
         ctx->pushCode( attribStep );
         ctx->pushCode( attrib->generator() );
      }
   }
}

void ModSpace::PStepLoader::apply_( const PStep* self, VMContext* ctx )
{
   static PStep* initStep = &Engine::instance()->stdSteps()->m_fillInstance;

   Error* error = 0;
   const ModSpace::PStepLoader* pstep = static_cast<const ModSpace::PStepLoader* >(self);

   int& seqId = ctx->currentCode().m_seqId;
   // the module we're working on is at top data stack.
   Module* mod = static_cast<Module*>(ctx->topData().asInst());
   ModSpace* ms = pstep->m_owner;

   bool isCalled = (seqId & 4) != 0;

   if( seqId < 0x10 )
   {
      TRACE( "ModSpace::PStepLoader::apply_ step 0 - %d on module %s", seqId, mod->name().c_ize() );

      bool isLoad = (seqId & 1) != 0;
      bool isMain = (seqId & 2) != 0;
      mod->setMain( isMain );
      mod->usingLoad( isLoad );

      seqId |= 0x10;
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
         ctx->pushData( Item() );
         ctx->pushCode( &ms->m_stepResolver );
         return;
      }
   }

   if( (seqId &0xF0) == 0x10 )
   {
      TRACE( "ModSpace::PStepLoader::apply_ step 10 on module %s", mod->name().c_ize() );
      seqId = (seqId&0xf) | 0x20;
      // Now that deps are resolved, do imports.
      ms->importInModule( mod, error );
      if( error != 0 ) {
         throw error;
      }
      ms->importInNS( mod );

      int32 icount = mod->getInitCount();
      if( icount != 0 )
      {
         // prepare all the required calls.

         for( int32 i = 0; i < icount; ++i )
         {
            Class* cls = mod->getInitClass(i);
            ctx->pushCode( initStep );
            ctx->callItem( Item(cls->handler(), cls) );
         }

         return;
      }
   }

   // fill attributes
   if( (seqId &0xF0) == 0x20 )
   {
      TRACE( "ModSpace::PStepLoader::apply_ step 11 on module %s", mod->name().c_ize() );
      seqId = (seqId&0xf) | 0x30;
      bool bDone = false;

      // check module attributes
      pushAttribs( ctx, mod->attributes(), bDone );


      // check mantra attributes
      Module::Private::MantraMap::iterator mi = mod->_p->m_mantras.begin();
      Module::Private::MantraMap::iterator me = mod->_p->m_mantras.end();
      while( mi != me ) {
         Mantra* mantra = mi->second;
         pushAttribs( ctx, mantra->attributes(), bDone );
         if( mantra->isCompatibleWith( Mantra::e_c_falconclass ) )
         {
            FalconClass* cls = static_cast<FalconClass*>(mantra);
            cls->registerAttributes( ctx );
         }
         ++mi;
      }

      // process all the expressions
      if( bDone ) {
         return;
      }
   }

   if( (seqId &0xF0) == 0x30 )
   {
      TRACE( "ModSpace::PStepLoader::apply_ step 12 on module %s", mod->name().c_ize() );
      seqId = (seqId&0xf) | 0x40;

      // Push the main funciton only if this is not a main module.
      if( mod->getMainFunction() != 0 && ! mod->isMain() )
      {
         static PStep* pop = &Engine::instance()->stdSteps()->m_pop;
         // we're pretty done
         // if directly called...
         if( isCalled )
         {
            // we don't want the module...
            ctx->popData();
            ctx->popCode();
         }
         else {
            // and if indirect, we don't want the main function result.
            ctx->resetCode(pop);
         }
         ctx->call( mod->getMainFunction() );
         return;
      }
   }

   // Third step -- we can remove self and the module.
   TRACE( "ModSpace::PStepLoader::apply_ step 13 (end) on module %s", mod->name().c_ize() );
   // leave the module in the stack if it's main.
   if( mod->isMain() )
   {
      // we need to return an extra reference to the process caller.
      mod->incref();
      ms->m_mainMod = mod;
   }
   else
   {
      //ctx->popData();
   }

   ctx->popCode();
}


void ModSpace::PStepResolver::apply_( const PStep* self, VMContext* ctx )
{
   const ModSpace::PStepResolver* pstep = static_cast<const ModSpace::PStepResolver* >(self);
   int& seqId = ctx->currentCode().m_seqId;

   // Here's the module that was previouously resolved (if any):
   Module* resolved = ctx->topData().isNil() ? 0 : static_cast<Module*>(ctx->topData().asInst());
   ctx->popData();

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
      if( resolved != 0 )
      {
         TRACE( "ModSpace::PStepResolver::apply_  on saving resolved module %s on request %d/%s of module %s",
                  resolved->name().c_ize(), seqId, idef->modReq()->name().c_ize(), mod->name().c_ize() );
         idef->modReq()->module( resolved );
         resolved = 0;
      }

      // do we have a module to import?
      if( ! idef->processed() )
      {
         if( idef->sourceModule().size() != 0 )
         {
            Module* other;

            if( idef->modReq()->module() == 0 )
            {
               // already available?
               other = ms->findModule( idef->sourceModule(), idef->isUri() );
               if( other == 0 )
               {
                  if( idef->loaded() ) {
                     // we should not be here
                     fassert2( false, "loaded but not resolved. ");
                     ++seqId;
                     continue;
                  }
                  idef->loaded(true);

                  // No? -- load it -- and always launch main
                  // -- keep current sequence ID
                  ms->loadModuleInContext( idef->sourceModule(), idef->isUri(), idef->isLoad(), false, ctx, mod, false );
                  // -- and wait for the process to have resolved it.
                  return;
               }

               idef->modReq()->module( other );
            }
            else {
               other = idef->modReq()->module();
            }

            idef->processed(true);

            if( idef->symbolCount() != 0 )
            {
               for( int i = 0; i < idef->symbolCount(); ++i )
               {
                  const String& symname = idef->sourceSymbol(i);
                  Item* orig = other->getGlobalValue(symname);
                  if( orig == 0 ) {
                     Error* em = new LinkError( ErrorParam(e_undef_sym, __LINE__, SRC )
                        .origin( ErrorParam::e_orig_linker )
                        .module(mod->uri())
                        .line( idef->sr().line() )
                        .extra( symname ) );

                     throw em;
                  }

                  String finalName = idef->targetSymbol(i);
                  mod->resolveExternValue(finalName, other, orig);
               }
            }

         }
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

      static PStep* pop = &Engine::instance()->stdSteps()->m_pop;
      ctx->pushCode(pop);
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
   static PStep* pop = &Engine::instance()->stdSteps()->m_pop;

   Module* mod = static_cast<Module*>( ctx->topData().asInst() );
   TRACE( "ModSpace::PStepExecMain::apply_  module \"%s\" found, resolving deps",
                  ctx->topData().asString()->c_ize() );

   mod->setMain(false);
   // we're called just once
   ctx->popCode();

   if( mod->getMainFunction() != 0 ) {
      ctx->pushCode( pop );
      ctx->call( mod->getMainFunction());
   }
}

void ModSpace::PStepStartLoad::apply_( const PStep* ps, VMContext* ctx )
{
   const PStepStartLoad* sl = static_cast<const PStepStartLoad*>(ps);

   Module* invoker = ctx->opcodeParam(1).isNil() ? 0 : static_cast<Module*>(ctx->opcodeParam(1).asInst());
   String modName = *ctx->opcodeParam(0).asString();
   int32 vals = ctx->currentCode().m_seqId;
   bool isUri = (vals & 1)!=0;
   bool asLoad = (vals & 2)!=0;
   bool isMain = (vals & 4)!=0;

   // we're called just once
   ctx->popCode();
   // I am not sure we actually need to pop. If not, we can use the modName by ref.
   ctx->popData(2);

   sl->m_owner->loadModuleInContext(modName, isUri, asLoad, isMain, ctx, invoker, true );
}

}

/* end of modspace.cpp */

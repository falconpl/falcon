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
#include <falcon/stderrors.h>
#include <falcon/itemarray.h>
#include <falcon/pseudofunc.h>
#include <falcon/process.h>
#include <falcon/vm.h>
#include <falcon/stdsteps.h>
#include <falcon/synfunc.h>
#include <falcon/psteps/stmtreturn.h>
#include <falcon/falconclass.h>
#include <falcon/stdhandlers.h>
#include <falcon/classes/classmodule.h>
#include <falcon/log.h>

#include <map>
#include <deque>
#include <list>

#include "module_private.h"

namespace Falcon
{

//========================================================================
// PSteps used by ModSpace
//========================================================================

class ModSpace::PStepManagedLoadedModule: public PStep
{
	public:
		PStepManagedLoadedModule( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepManagedLoadedModule() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepManagedLoadedModule";
		}

	private:
		static void apply_( const PStep* self, VMContext* ctx );
		ModSpace* m_owner;
};

class ModSpace::PStepResolveModReq: public PStep
{
	public:
		PStepResolveModReq( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepResolveModReq() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepResolveModReq";
		}

	private:
		static void apply_( const PStep* self, VMContext* ctx );
		ModSpace* m_owner;
};

class ModSpace::PStepStoreModReq: public PStep
{
	public:
		PStepStoreModReq( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepStoreModReq() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepStoreModReq";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );

};

class ModSpace::PStepSetupModule: public PStep
{
	public:
		PStepSetupModule( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepSetupModule() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepSetupModule";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );

};

class ModSpace::PStepCompleteLoad: public PStep
{
	public:
		PStepCompleteLoad( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepCompleteLoad() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepCompleteLoad";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );
};


class ModSpace::PStepDisposeLoad: public PStep
{
	public:
		PStepDisposeLoad( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepDisposeLoad() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepDisposeLoad";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );
};

class ModSpace::PStepSaveDynMantra: public PStep
{
	public:
		PStepSaveDynMantra( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepSaveDynMantra() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepSaveDynMantra";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );
};

class ModSpace::PStepSetProcResult: public PStep
{
	public:
		PStepSetProcResult( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepSetProcResult() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepSetProcResult";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );
};


class ModSpace::PStepCallMain: public PStep
{
	public:
		PStepCallMain( ModSpace* owner ): m_owner( owner )
		{
			apply = apply_;
		}
		virtual ~PStepCallMain() {};
		virtual void describeTo( String& str ) const
		{
			str = "PStepCallMain";
		}

		//Need to do something about this
        ModSpace* m_owner;

	private:
		static void apply_( const PStep* self, VMContext* ctx );
};


void ModSpace::PStepManagedLoadedModule::apply_( const PStep* self, VMContext* ctx )
{
	Error* error = 0;
	const ModSpace::PStepManagedLoadedModule* pstep = static_cast<const ModSpace::PStepManagedLoadedModule* >( self );

	CodeFrame& current = ctx->currentCode();
	int& seqId = current.m_seqId;
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );
	ModSpace* ms = pstep->m_owner;

	TRACE( "ModSpace::PStepManagedLoadedModule::apply_ - %d on module %s", seqId, mod->name().c_ize() );

	// we don't need to be around anymore
	ctx->popCode();
	// the code below us should be PStepSetupModule

	bool isLoad = ( seqId & 1 ) != 0;
	bool isMain = ( seqId & 2 ) != 0;
	mod->setMain( isMain );
	mod->usingLoad( isLoad );

	// store the module
	ms->store( mod );

	// perform exports, so imports can find them on need.
	if( isLoad )
	{
		ms->exportFromModule( mod, error );

		if( error != 0 )
		{
			throw error;
		}
	}

	//do we have some dependency?
	if( mod->modRequestCount() != 0 )
	{
		// let's manage them.
		ctx->stepIn( ms->m_stepResolveModReq );
	}
}


void ModSpace::PStepResolveModReq::apply_( const PStep* self, VMContext* ctx )
{
	const ModSpace::PStepResolveModReq* pstep = static_cast<const ModSpace::PStepResolveModReq* >( self );

	CodeFrame& current = ctx->currentCode();
	int& seqId = current.m_seqId;
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );
	ModSpace* ms = pstep->m_owner;

	TRACE( "ModSpace::PStepResolveModReq::apply_ - step %d on module %s", seqId, mod->name().c_ize() );

	int depCount = ( int ) mod->modRequestCount();
	while( seqId < depCount )
	{
		// get the required dependency
		ModRequest* mreq = mod->getModRequest( seqId );

		TRACE1( "ModSpace::PStepResolveModReq::apply_ - Checking dep %d: %s(%s %s)%s",
		        seqId,
		        mreq->name().c_ize(),
		        ( mreq->isUri() ? "URI" : "path" ),
		        ( mreq->isLoad() ? "load" : "import" ),
		        ( mreq->module() == 0 ? "" : " already resolved" )
		      );

		// prepare for next loop
		++seqId;

		// still unresolved?
		if( mreq->module() == 0 )
		{
			// are we lucky?
			Module* found = 0;
			if( mreq->isUri() )
			{
				found = ms->findByURI( mreq->name() );
			}
			else
			{
				found = ms->findByName( mreq->name() );
			}

			if( found != 0 )
			{
				if( found == mod )
				{
					throw FALCON_SIGN_XERROR( LinkError, e_mod_load_self, .extra( mreq->name() ).origin( ErrorParam::e_orig_loader ) );
				}
				mreq->module( found );
				mod->onModuleResolved( mreq );
			}
			else
			{
				// start the load process.
				ctx->pushCode( ms->m_stepStoreModReq );
				ctx->currentCode().m_seqId = seqId - 1;
				ms->loadModuleInContext( mreq->name(), mreq->isUri(), mreq->isLoad(), false, ctx, mod, true );
				return;
			}
		}
	}

	// we did our job.
	ctx->popCode();
}


void ModSpace::PStepStoreModReq::apply_( const PStep*, VMContext* ctx )
{
	CodeFrame& current = ctx->currentCode();
	int& seqId = current.m_seqId;
	// the module we're working on is at top data stack.
	Module* loadedMod = static_cast<Module*>( ctx->topData().asInst() );
	ctx->popData();
	Module* sourceMod = static_cast<Module*>( ctx->topData().asInst() );

	TRACE( "ModSpace::PStepResolveModReq::apply_ - step %d on module %s <- %s ", seqId,
	       sourceMod->name().c_ize(), loadedMod->name().c_ize() );

	// save in the correct position.
	ModRequest* req = sourceMod->getModRequest( seqId );
	fassert( req != 0 );
	req->module( loadedMod );
	// abandon the extra reference.
	loadedMod->decref();
	// signal the loader about the news
	sourceMod->onModuleResolved( req );

	// we're not needed anymore.
	ctx->popCode();
}


void ModSpace::PStepSetupModule::apply_( const PStep*, VMContext* ctx )
{
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );

	TRACE( "ModSpace::PStepSetupModule::apply_ - module %s ", mod->name().c_ize() );

	// we're not needed anymore.
	ctx->popCode();

	// resolve all the import defs.
	Error* error = 0;
	if( ! mod->resolveImports( error ) )
	{
		throw error;
	}

	// Give the module the good news.
	mod->onLinkComplete( ctx );

	// Eventually ready the main functions.
	if( ! mod->isMain() && mod->getMainFunction() != 0 )
	{
		ctx->pushCode( mod->modSpace()->m_stepCallMain );
	}

	// prepare the inits of this module
	mod->startup( ctx );

	// on exit, the init routines + the main function will be executed as needed.
}


void ModSpace::PStepCallMain::apply_( const PStep*, VMContext* ctx )
{
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );

	TRACE( "ModSpace::PStepCallMain::apply_ - module %s ", mod->name().c_ize() );

	// Tentatively, the process result is the result of this function.
	// this will also pop the function result, as we don't want it around.
	ctx->resetCode( mod->modSpace()->m_stepSetProcResult );
	ctx->call( mod->getMainFunction() );
	// the main function will be executed as needed.
}


void ModSpace::PStepDisposeLoad::apply_( const PStep*, VMContext* ctx )
{
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );

	TRACE( "ModSpace::PStepDisposeLoad::apply_ - module %s ", mod->name().c_ize() );

	// we're not needed anymore.
	ctx->popCode();

	// Give the module the good news.
	mod->onStartupComplete( ctx );

	// remove the module from the stack.
	ctx->popData();
	// but don't decref: the reference is held by the GC.
}

void ModSpace::PStepCompleteLoad::apply_( const PStep*, VMContext* ctx )
{
	// the module we're working on is at top data stack.
	Module* mod = static_cast<Module*>( ctx->topData().asInst() );

	TRACE( "ModSpace::PStepCompleteLoad::apply_ - module %s ", mod->name().c_ize() );

	// add a reference, as it will be asked from outside.
	mod->incref();

	// we're not needed anymore.
	ctx->popCode();

	// Give the module the good news.
	mod->onStartupComplete( ctx );

	// keep the module in the stack, something else needs it.
}



void ModSpace::PStepSaveDynMantra::apply_( const PStep* self, VMContext* ctx )
{
	static Engine* eng = Engine::instance();
	Mantra* theClass;

	ModSpace* ms = static_cast<const ModSpace::PStepSaveDynMantra*>( self )->m_owner;
	// get the parameters
	Module* clsContainer = ctx->topData().isNil() ? 0 : static_cast<Module*>( ctx->topData().asInst() );
	const String& className = *ctx->opcodeParam( 1 ).asString();
	TRACE( "ModSpace::PStepSaveDynMantra::apply_ - search \"%s\" in %s -- depth %d",
	       className.c_ize(), clsContainer == 0 ? "<nothing>" : clsContainer->uri().c_ize(), ( int ) ctx->dataSize() );

	// remove the module and the class name
	ctx->popData( 2 );

	// still no luck?
	if( clsContainer == 0 )
	{
		Item* sym = ms->findExportedValue( className );
		if( sym == 0 )
		{
			theClass = eng->getMantra( className );
			if( theClass == 0 )
			{
				throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
				                               .origin( ErrorParam::e_orig_runtime )
				                               .extra( "Unavailable mantra " + className ) );
			}
		}
		else
		{

			if( sym->isClass() || sym->isFunction() )
			{
				theClass = static_cast<Mantra*>( sym->asInst() );
			}
			else
			{
				throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
				                               .origin( ErrorParam::e_orig_runtime )
				                               .extra( "Symbol is not mantra " + className ) );
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
			clsContainer->decref();
			throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
			                               .origin( ErrorParam::e_orig_runtime )
			                               .extra( expl ) );
		}

		clsContainer->decref();
	}

	ctx->pushData( Item( theClass->handler(), theClass ) );
	ctx->popCode();
}


void ModSpace::PStepSetProcResult::apply_( const PStep*, VMContext* ctx )
{
	int32 seqId = ctx->currentCode().m_seqId;
	TRACE( "ModSpace::PStepSetProcResult::apply_ - mode %s ", seqId == 0 ? "Save" : "Restore" );

	// needed just once.
	ctx->popCode();

	if( seqId == 0 )
	{
		ctx->process()->setResult( ctx->topData() );
		ctx->popData();
	}
	else
	{
		ctx->pushData( ctx->process()->result() );
	}
}


//========================================================================
// Main ModSpace class
//========================================================================
Class* ModSpace::handler()
{
	static Class* ms = Engine::handlers()->modSpaceClass();
	return ms;
}

class ModSpace::Private
{
	public:
		typedef std::map<const Symbol*, Item*> ExportSymMap;
		ExportSymMap m_symMap;
		std::list<Item> m_privateItems;

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
				Module* mod = iter->second;
				// Remove the weak reference with us.
				ModSpace* owner = mod->modSpace();
				mod->modSpace( 0 );
				mod->onRemoved( owner );
				mod->decref();
				++iter;
			}

			ExportSymMap::iterator esi = m_symMap.begin();
			while( esi != m_symMap.end() )
			{
				esi->first->decref();
				++esi;
			}
		}
};

//==============================================================
// Main class
//

ModSpace::ModSpace( Process* owner, ModSpace* parent ):
	_p( new Private ),
	m_parent( parent ),
	m_lastGCMark( 0 ),
	m_mainMod( 0 ),
	m_loadModuleCbFunc( 0 )
{
	m_process = owner;
	m_loader = new ModLoader( this );
	SynFunc* sf = new SynFunc( "$loadModule" );
	sf->syntree().append( new StmtReturn );
	m_loaderFunc = sf;

	m_stepManagedLoadedModule = new PStepManagedLoadedModule( this );
	m_stepResolveModReq = new PStepResolveModReq( this );
	m_stepStoreModReq = new PStepStoreModReq( this );
	m_stepSetupModule = new PStepSetupModule( this );
	m_stepCompleteLoad = new PStepCompleteLoad( this );
	m_stepDisposeLoad = new PStepDisposeLoad( this );
	m_stepSaveDynMantra = new PStepSaveDynMantra( this );
	m_stepSetProcResult = new PStepSetProcResult( this );
	m_stepCallMain = new PStepCallMain( this );

	if( parent != 0 )
	{
		parent->incref();
	}
}


ModSpace::~ModSpace()
{
	TRACE( "ModSpace::~ModSpace deleting %p", this );
	if( m_parent != 0 )
	{
		m_parent->decref();
	}

	delete _p;
	delete m_loader;
	delete m_loaderFunc;

	delete m_stepManagedLoadedModule;
	delete m_stepResolveModReq;
	delete m_stepStoreModReq;
	delete m_stepSetupModule;
	delete m_stepCompleteLoad;
	delete m_stepDisposeLoad;
	delete m_stepSaveDynMantra;
	delete m_stepSetProcResult;
	delete m_stepCallMain;

	TRACE( "ModSpace::~ModSpace complete deletion of %p", this );
}


void ModSpace::setParent( ModSpace* parent )
{
	parent->incref();
	if( m_parent != 0 )
	{
		m_parent->decref();
	}
	m_parent = parent;
}


Process* ModSpace::loadModule( const String& name, bool isUri,  bool asLoad, bool isMain )
{
	TRACE( "ModSpace::loadModule(%s, %s, %s, %s )",
	       name.c_ize(), ( isUri ? "uri" : "name" ), ( asLoad ? "load" : "import" ), ( isMain ? "main" : "normal" ) );
	Process* process = m_process->vm()->createProcess();
	loadModuleInProcess( process, name, isUri, asLoad, isMain );
	return process;
}

Process* ModSpace::loadModule( const String& name, Stream* script, const String& path, ModLoader::t_modtype type )
{
	Process* process = m_process->vm()->createProcess();

	process->adoptModSpace( this );

	VMContext* tgtContext = process->mainContext();
	tgtContext->call( m_loaderFunc );
	tgtContext->pushCode( &Engine::instance()->stdSteps()->m_returnFrameWithTop );

	tgtContext->pushCode( m_stepCompleteLoad );

	// push the code that will invoke module setup on success.
	tgtContext->pushCode( m_stepSetupModule );

	// push the code that will handle the load result on success.
	// this will also push the resolver step if needed.
	tgtContext->pushCode( m_stepManagedLoadedModule );

	// we need to know what we should do with the module.
	tgtContext->currentCode().m_seqId = 1 | 2;

	// let loadFile to compute the module  name.
	bool loading = m_loader->loadMem( tgtContext, name, script, path, type );

	if( ! loading )
	{
		throw new IOError( ErrorParam( e_mod_notfound, __LINE__, SRC )
		                   .module( "" )
		                   .extra( name + " in " + m_loader->getSearchPath() ) );
	}

	return process;
}

bool ModSpace::loadModule( VMContext* tgtctx, const String& name )
{
    if( m_loadModuleCbFunc )
    return m_loadModuleCbFunc( tgtctx, name );

    return false;
}

void ModSpace::loadModuleInProcess( Process* process, const String& name, bool isUri,  bool asLoad, bool isMain, Module* loader )
{
	process->adoptModSpace( this );

	VMContext* tgtContext = process->mainContext();
	tgtContext->call( m_loaderFunc );
	tgtContext->pushCode( &Engine::instance()->stdSteps()->m_returnFrameWithTop );

	loadModuleInContext( name, isUri, asLoad, isMain, tgtContext, loader );
}


void ModSpace::loadAndRun( Process* process, const String& name, bool isUri, Module* loader )
{
	process->adoptModSpace( this );

	VMContext* tgtContext = process->mainContext();
	tgtContext->call( m_loaderFunc );

	// the restored value will be the return value -- aka the process value
	tgtContext->pushCode( &Engine::instance()->stdSteps()->m_returnFrameWithTop );

	// push a "restore process value" pstep at bottom of the context.
	tgtContext->pushCode( m_stepSetProcResult );
	tgtContext->currentCode().m_seqId = 1;

	loadModuleInContext( name, isUri, false, false, tgtContext, loader );
}


void ModSpace::loadModuleInProcess( const String& name, bool isUri, bool asLoad, bool isMain, Module* loader )
{
	loadModuleInProcess( m_process, name, isUri, asLoad, isMain, loader );
}


void ModSpace::loadModuleInContext( const String& name, bool isUri, bool isLoad, bool isMain, VMContext* tgtContext, Module* loader, bool isNeeded )
{
	TRACE1( "ModSpace::loadModuleInContext(%s, %s, %s, %s, %p/%d, %s )",
	        name.c_ize(), ( isUri ? "uri" : "name" ), ( isLoad ? "load" : "import" ), ( isMain ? "main" : "normal" ),
	        tgtContext, tgtContext->id(), ( loader == 0 ? "no loader" : loader->name().c_ize() ) );

	// determine the fate of the loaded module.
	if( isMain || isNeeded )
	{
		tgtContext->pushCode( m_stepCompleteLoad );
	}
	else
	{
		tgtContext->pushCode( m_stepDisposeLoad );
	}

	// push the code that will invoke module setup on success.
	tgtContext->pushCode( m_stepSetupModule );

	// push the code that will handle the load result on success.
	// this will also push the resolver step if needed.
	tgtContext->pushCode( m_stepManagedLoadedModule );

	// we need to know what we should do with the module.
	tgtContext->currentCode().m_seqId = ( isLoad ? 1 : 0 ) | ( isMain ? 2 : 0 );

	// start the loading process.
	bool loading;
	if( isUri )
	{
		// let loadFile to compute the module  name.
		loading = m_loader->loadFile( tgtContext, "", name, ModLoader::e_mt_none, true, loader );
	}
	else
	{
		loading = m_loader->loadName( tgtContext, name, ModLoader::e_mt_none, loader );
	}

	if( ! loading && tgtContext->topData().isNil() )
	{
		throw new IOError( ErrorParam( e_mod_notfound, __LINE__, SRC )
		                   .module( loader == 0 ? "" : loader->name() )
		                   .extra( name + " in " + m_loader->getSearchPath() ) );
	}

	// the loader will have pushed the needed steps to complete the operation, by now.
	// the process will continue at PStepManagedLoadedModule::apply_.
}


void ModSpace::add( Module* mod )
{
	TRACE( "ModSpace::add %s", mod->name().c_ize() );

	store( mod );

	Error* le = 0;
	exportFromModule( mod, le );
	if( le != 0 )
	{
		throw le;
	}
}

void ModSpace::add( loadModule_cbfunction func )
{
    m_loadModuleCbFunc = func;
}


void ModSpace::store( Module* mod )
{
	TRACE( "ModSpace::store %s", mod->name().c_ize() );

	Engine::instance()->log()->log( Log::fac_engine, Log::lvl_info,
	                                String( "Adding module " ).A( mod->name() ).A( "(" ).A( mod->uri() ).A( ")" ) );

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
			name.N( count );
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
	mod->modSpace( this );

	// invoke the callback to inform the module.
	mod->onLoad();

	// invoke the process information handler
	m_process->onModuleAdded( mod );
}


void ModSpace::resolveDeps( VMContext* ctx, Module* mod )
{
	static Class* modClass = Engine::instance()->stdHandlers()->moduleClass();

	TRACE( "ModSpace::resolveDeps on module %s, %d deps", mod->name().c_ize(), mod->modRequestCount() );

	if( mod->modRequestCount() > 0 )
	{
		mod->incref();
		ctx->pushData( Item(modClass, mod) );
		ctx->pushCode( m_stepDisposeLoad );
		ctx->pushCode( m_stepResolveModReq );
	}
}


bool ModSpace::exportFromModule( Module* mod, Error*& link_errors )
{
	TRACE1( "ModSpace::exportFromModule %s", mod->name().c_ize() );

	class Rator: public GlobalsMap::VariableEnumerator
	{
		public:
			Rator( ModSpace* owner, Module* mod, Error*& link_errors ):
				m_owner( owner ),
				m_module( mod ),
				m_link_errors( link_errors )
			{}
			virtual ~Rator() {};

			virtual void operator()( const Symbol* sym, Item*& value )
			{
				bool status =  m_owner->exportSymbol( sym, value );
				if( ! status )
				{
					Error* e = new LinkError( ErrorParam( e_already_def )
					                          .origin( ErrorParam::e_orig_linker )
					                          .module( m_module->uri() )
					                          .extra( sym->name() ) );

					m_owner->addLinkError( m_link_errors, e );
				}

			}

		private:
			ModSpace* m_owner;
			Module* m_module;
			Error*& m_link_errors;
	}
	rator( this, mod, link_errors );

	mod->globals().enumerateExports( rator );
	return link_errors == 0;
}


bool ModSpace::exportSymbol( const Symbol* sym, Item* value )
{
	TRACE1( "ModSpace::exportSymbol %s", sym->name().c_ize() );

	Private::ExportSymMap::iterator iter = _p->m_symMap.find( sym );
	if( iter != _p->m_symMap.end() )
	{
		// name clash!
		TRACE1( "ModSpace::exportSymbol %s defined in this space, failing.", sym->name().c_ize() );
		return false;
	}

	// link the symbol
	TRACE1( "ModSpace::exportSymbol exporting %s", sym->name().c_ize() );

	sym->incref();
	_p->m_symMap.insert( std::make_pair( sym, value ) );
	return true;
}


void ModSpace::setExportValue( const String& symName, const Item& value )
{
	const Symbol* sym = Engine::getSymbol( symName );
	setExportValue( sym, value );
	sym->decref();
}



void ModSpace::setExportValue( const Symbol* sym, const Item& value )
{
	TRACE1( "ModSpace::setExportValue %s", sym->name().c_ize() );

	Private::ExportSymMap::iterator iter = _p->m_symMap.find( sym );
	if( iter != _p->m_symMap.end() )
	{
		// name clash!
		TRACE1( "ModSpace::exportSymbol %s defined in this space.", sym->name().c_ize() );
		iter->second->copyFromLocal( value );
	}
	else
	{
		// link the symbol
		TRACE1( "ModSpace::exportSymbol exporting %s", sym->name().c_ize() );

		sym->incref();
		Item* slot = getNewGlobalSlot( value );
		_p->m_symMap.insert( std::make_pair( sym, slot ) );
	}
}


Item* ModSpace::getNewGlobalSlot( const Item& value )
{
	_p->m_privateItems.push_back( value );
	Item& back = _p->m_privateItems.back();
	return &back;
}

void ModSpace::gcMark( uint32 mark )
{
	if( m_lastGCMark != mark )
	{
		m_lastGCMark = mark;
		Private::ExportSymMap::iterator esi = _p->m_symMap.begin();
		Private::ExportSymMap::iterator esi_end = _p->m_symMap.end();

		while( esi != esi_end )
		{
			Item* item = esi->second;
			item->gcMark( mark );
			++esi;
		}

		Private::ModMap::iterator mmi = _p->m_modmap.begin();
		Private::ModMap::iterator mmi_end = _p->m_modmap.end();

		while( mmi != mmi_end )
		{
			mmi->second->gcMark( mark );
			++mmi;
		}
	}
}


Item* ModSpace::findExportedValue( const Symbol* sym )
{
	Private::ExportSymMap::iterator iter = _p->m_symMap.find( sym );

	if( iter != _p->m_symMap.end() )
	{
		Item* value = iter->second;
		return value;
	}

	if( m_parent != 0 )
	{
		return m_parent->findExportedValue( sym );
	}

	return 0;
}


Item* ModSpace::findExportedValue( const String& symName )
{
	const Symbol* sym = Engine::getSymbol( symName );
	Item* value = findExportedValue( sym );
	sym->decref();
	return value;
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
	if( top == 0 )
	{
		top = new GenericError( ErrorParam( e_link_error, __LINE__, SRC ) );
	}

	top->appendSubError( newError );
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
		// we need the module we just found -- so save it
		ctx->pushCode( m_stepCompleteLoad );

		// push the code that will invoke module setup on success.
		ctx->pushCode( m_stepSetupModule );

		// push the code that will handle the load result on success.
		// this will also push the resolver step if needed.
		ctx->pushCode( m_stepManagedLoadedModule );

		// we need to know what we should do with the module.
		ctx->currentCode().m_seqId = 4; // non-load, non-main, get result

		// start the loading process.
		bool loading;
		loading = m_loader->loadFile( ctx, moduleName, moduleUri, ModLoader::e_mt_none, true );
		if( ! loading )
		{
			ctx->popData();
			m_loader->loadName( ctx, moduleName, ModLoader::e_mt_none );
			// in case of failure, a Item() is already pushed by loadName.
		}
	}
	else
	{
		TRACE( "ModSpace::retreiveDynamicModule found module %s: %s",
		       clsContainer->name().c_ize(), clsContainer->uri().c_ize() );
		// the gc/caller will decref this.
		clsContainer->incref();
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
		Mantra* result = eng->getMantra( className );
		if( result != 0 )
		{
			MESSAGE( "Found required mantra in engine." );
			ctx->pushData( Item( result->handler(), result ) );
			return;
		}
		else
		{
			// we can't find this stuff in modules -- as we don't have a module to search for.
			ctx->pushData( Item() );
			return;
		}
	}

	// push the mantra resolver step.
	ctx->pushData( FALCON_GC_HANDLE( new String( className ) ) );
	ctx->pushCode( m_stepSaveDynMantra );

	// and then start resolving the module
	retreiveDynamicModule( ctx, moduleUri, moduleName );
}


void ModSpace::enumerateIStrings( IStringEnumerator& cb ) const
{
	Private::ModMap::const_iterator miter = _p->m_modmap.begin();
	while( miter != _p->m_modmap.end() )
	{
		miter->second->enumerateIStrings( cb );
		++miter;
	}

	if( m_parent != 0 )
	{
		m_parent->enumerateIStrings( cb );
	}
}

}

/* end of modspace.cpp */

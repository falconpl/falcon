/*
   FALCON - The Falcon Programming Language.
   FILE: modspace.h

   Module space for the Falcon virtual machine
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MODSPACE_H
#define FALCON_MODSPACE_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/modmap.h>
#include <falcon/pstep.h>

#include <falcon/refcounter.h>
#include <falcon/enumerator.h>

#include <falcon/stream.h>
#include <falcon/modloader.h>

namespace Falcon
{

class VMContext;
class Module;
class Error;
class Symbol;
class ModLoader;
class ImportDef;
class Mantra;
class Process;
class Function;
class Item;

/** Collection of (static) modules active in a virtual machine.

 The module place is a place where modules reside and possibly share their
 data through the standard import-export directives.

 Steps to get a module ready to run:

 # add() (optional) predefined modules not needing to be resolved.
 # resolve() - resolves all the dependencies eventually loading what's needed.
 # link() - publishes exports resolves imports from each module
 # apply() - stores the main methods of each module in the VM in proper order.

 Then just run the VM.

 The process can be repeated more than once for the same ModSpace, calling
 add() and/or resolve() multiple times before each link() and apply(). Also,
 link() can be called multiple times before apply().

 This step functions will operate only if there is actually something to do for
 them. For instance, adding a module that is already present won't have any
 effect; calling link() will just resolve the newly pending dependencies that
 have been introduced since last call; calling apply() will add to the VM
 just the __main__ functions of the newly read modules since last apply()
 was called and so on.

 Children ModSpaces can be used to put inside them an insolated set of modules.
 They will look into parent modules for already resolved modules and as a place
 where to copy public symbols.

 \note Explicit compilation requests (via compile from scratch and include())
 might create a new module space that might be eventually linked with the module
 space of the module where the compilation takes place. This relationship
 allows to decide whether the compilation is dynamic, static or partial.
 # If the module is added to the loading/compiling module space, every module
   included by the loaded module is also stored in that ModSpace.
 # If the moduled is added to a new ModSpace which has the loaders module as
   a parent ModSpace, existing modules can be used by the loaded one, but other
   modules needed by the loaded one are loaded anew and added to the new space.
 # If the module is added to a new ModSpace without any parent, the module is
   practically living in a separate space and needs to load anew any dependency
   (including the Core Module).
*/
class FALCON_DYN_CLASS ModSpace
{

	public:
		/** Creates the module space, eventually linked to a parent module space.
		 \param owner The virtual machine owning this module space.
		 \param parent A parent module space where to look for already loaded modules
		 or exported symbols.
		 */
		ModSpace( Process* owner, ModSpace* parent = 0 );

		/** Adds a new module (internal) to the module space.
		 \param module The module to be added.

		 This method is mainly thought to add a module that has been
		 created by a third party application in a module space, for
		 it to be usable by new code incoming later on.

		 If the same identical module is present (by address), the method returns
		 false. As multiple modules having the same logical name should not exist in
		 the same space, the incoming module will automatically receive a different
		 unique name (schemed after "modname-X", where X is an incremental number).
		 If this is not desirable, check the result of getModuleByName() method
		 before trying to add the module.

		 The module exported symbols are
		 imported in the module space. This might throw an "already defined" error.
		 In this case, the error is thrown after the module is already added to the
		 module space.

		 \note add() is actually store() + forced export.
		 */
		void add( Module* mod );

		/** Store a module without exporting its values.

		    If the module is stored for load the caller should then call
		    exportFromModule() autonomously.
		 */
		void store( Module* mod );

		/** Creates a process ready to load the module and all its dependencies.
		 * \param name The name of the module to be loaded.
		 * \param isUri true if the given name is actually a URI
		 * \param isMain true if the given module is considered the main module for the calling application.
		 *
		 * The returned process can be started immediately and then waited for completion.
		 * As the process is complete, the module and all its dependencies are loaded as required
		 * in this module space, or an error would be thrown.
		 *
		 * The following skeleton code can be used:
		 * \code
		      Process* prc = theSpace->loadModule( path, isFsPath, false );

		      try {
		         prc->start();
		         prc->wait(-1);
		      }
		      catch( Error* e ) {
		         // handle error e
		         e->decref();
		      }

		      prc->decref();
		 * \endcode
		 *
		 * If you're in need of loading a module from within a VM execution,
		 * you can still use this method if you want to do this asynchronously,
		 * but you can use loadModuleInContext() to use your own execution context
		 * instead of creating a new process.
		 */
		Process* loadModule( const String& name, bool isUri, bool asLoad, bool isMain = false );
		Process* loadModule( const String& name, Stream* script, const String& path = "", ModLoader::t_modtype type = ModLoader::e_mt_source );

		void loadModuleInProcess( const String& name, bool isUri, bool asLoad, bool isMain = false, Module* loader = 0 );
		void loadModuleInProcess( Process* prc, const String& name, bool isUri, bool asLoad, bool isMain = false, Module* loader = 0 );

		void loadModuleInContext( const String& name, bool isUri, bool asLoad, bool isMain, VMContext* tgtContext, Module* caller, bool isNeeded = false );

		/** Loads and runs a module in the given process.
		 * The module will be loaded, and eventually prepared for execution (if it provides a main function).
		 *
		 * The result of the process after a start/wait() pair will be the module main function return value.
		 *
		 * The module will be invisible outside the call, as it will be not ready before start and destroyed
		 * after wait.
		 *
		 */
		void loadAndRun( Process* process, const String& name, bool isUri, Module* loader = 0 );

		//===================================================================
		// Service functions
		//===================================================================

		void findDynamicMantra(
		    VMContext* ctx,
		    const String& moduleUri,
		    const String& moduleName,
		    const String& className );


		void gcMark( uint32 mark );

		uint32 currentMark() const
		{
			return m_lastGCMark;
		}


		/** Get the space in which this group resides. */
		ModSpace* parent() const
		{
			return m_parent;
		}


		/** Finds a module stored in this space by name.
		 \param name The name of the module.
		 \return A module if found, 0 otherwise.

		 This method searches a module in this space,  or in the parent space(s) if there are parents.
		 */
		inline Module* findByName( const String& name ) const
		{
			return findModule( name, false );
		}


		/** Finds a module stored in this space by its URI.
		 \param name The URI of the module.
		 \return A module if found, 0 otherwise.

		 This method searches a module in this space,  or in the parent space(s) if there are parents.
		 */
		inline Module* findByURI( const String& uri ) const
		{
			return findModule( uri, true );
		}

		/** Finds a module stored in this space by its name or URI.
		 \param name_uri The name or URI of the module.
		 \param isUri True if the name_uri parameter is a URI, false if it's a name.
		 \return A module if found, 0 otherwise.

		 This method searches a module in this space,
		 or in the parent space(s) if there are parents.
		 */
		Module* findModule( const String& name_uri, bool isUri ) const;


		/** Exports a single symbol on the module space.
		 * \param sym The symbol to be exported.
		 * \param value The value associated with the exported symbol.
		 * \return A pointer to the value entry created in the export table on success,
		 *    0 if the symbol is already exported in the module space.
		 *
		 * On success, the exporter should discard the item it was previously associated
		 * with the symbol, and use the returned value instead.
		 */
		bool exportSymbol( const Symbol* sym, Item* value );

		void setExportValue( const String& symName, const Item& value );
		void setExportValue( const Symbol* sym, const Item& value );

		/** Finds a value that has been generally exported via the load/export constructs.
		 \param symName the name of the global variable to be searched.
		 \return A pointer to the exported value, or 0 if not found.
		 */
		Item* findExportedValue( const String& symName );

		/** Finds a value that has been generally exported via the load/export constructs.
		 \param sym the name of the global variable to be searched.
		 \return A pointer to the exported value, or 0 if not found.
		 */
		Item* findExportedValue( const Symbol* sym );

		/** Gets the module loader associated with this virtual machine. */
		ModLoader* modLoader() const
		{
			return m_loader;
		}

		/** Virtual machine associated with this space. */
		Process* process() const
		{
			return m_process;
		}

		/**
		 * Returns the engine-wide visible instance of ClassModSpace;
		 */
		static Class* handler();

		Module* mainModule() const
		{
			return m_mainMod;
		}

		typedef Enumerator<String> IStringEnumerator;

		/** Enumerates all the international strings recorded by modules in this modulespace.
		 *
		 * Strings are not consolidated prior enumeration; this means the strings
		 * will be out of order, and there might be some duplicates.
		 *
		 */
		void enumerateIStrings( IStringEnumerator& is ) const;

		/** Adds this child modspace to the given parent.
		 When a child is added, the parent gets a new reference count.
		 When a child is destroyed, the parent gets dereferenced.

		   This means a modspace cannot be destroyed while it has children
		   still alive, but the children can be removed one by one without
		   the parent being notified (except for the decref).
		*/
		void setParent( ModSpace* parent );

		/** Prepare a module dependency resolver in the target context. */
		void resolveDeps( VMContext* ctx, Module* mod );

	private:
		class Private;
		ModSpace::Private* _p;
		friend class Private;

		Process* m_process;
		ModSpace* m_parent;
		uint32 m_lastGCMark;
		Function* m_loaderFunc;
		Module* m_mainMod;

		ModLoader* m_loader;

		Item* getNewGlobalSlot( const Item& value );

		virtual ~ModSpace();
		bool exportFromModule( Module* mod, Error*& link_errors );

		void addLinkError( Error*& top, Error* newError );

		void importSpecificDep( Module* asker, void* def, Error*& link_errors );
		void importGeneralDep( Module* asker, void* def, Error*& link_errors );


		void retreiveDynamicModule(
		    VMContext* ctx,
		    const String& moduleUri,
		    const String& moduleName );

		/** Gets the module that is on top of the stack and manages it.
		 *
		 * Operations performed by this step:
		 * # Export symbols if required.
		 * # Add module to the module space.
		 * # Prepare all the ModReq resolutions, pushing them in the stack.
		 */
		PStep* m_stepManagedLoadedModule;

		/** Resolves each module request.
		 *
		 * Gets a module request on top of the stack and processes it. The seqId is the
		 * count of module requests still left in the stack to be processed. When it's 0,
		 * the opcode can pop itself.
		 *
		 * A module request can be resolved either by finding the module in the ModSpace where
		 * the owner module resides, or in any parent ModSpace, or by initiating a new load
		 * in the current ModSpace.
		 */
		PStep* m_stepResolveModReq;

		/** Saves a just resolved module request.
		 *
		 * The topmost entity in the stack is the module that has been just loaded,
		 * while the second-topmost item is the loader (requester) module.
		 *
		 * This step pops the topmost item, with the freshly resolved module in it,
		 * and stores it in the nth ModRequest* entry of the loader module. The id
		 * of the host mod request waiting for the given module to be resolved
		 * is stored as seqId of this pstep.
		 */
		PStep* m_stepStoreModReq;

		/** Setup the module after all the module requests have been solved.
		 *
		 * This step performs the following operations:
		 * # invoke onLinkComplete() to inform the loaded module about all
		 *   dependencies having been resolved.
		 * # if required, invokes the main() function of the module, without
		 *   running it.
		 * # invokes the initModule() method to ask for the module to prepare
		 *   the VMContext as needed.
		 *
		 * The default action of initModule are that of creating the startup
		 * environment for the module, which usually translates in creating the
		 * falcon classes and preparing the init operations for static objects
		 * in the host context.
		 *
		 * When this pstep exits, the code stack will have the required init/main
		 * functions properly pushed to be run in order.
		 */
		PStep* m_stepSetupModule;

		/** Saves the module on top of the stack to the process result.
		 *
		 * This step invokes the module onStartupComplete() to inform the
		 * module that everything is green/go.
		 *
		 * The module is left on top of the data stack. Other psteps
		 * shall use/dispose of it.
		 *
		 * This step is pushed when the request to load a module is explicit,
		 * and the caller wants to have a reference to the module as a final
		 * result of the operation.
		 */
		PStep* m_stepCompleteLoad;

		/** Discards the module on top of the stack, as it's referenced elsewhere.
		 *
		 * This step invokes the module onStartupComplete() to inform the
		 * module that everything is green/go, and then pops the module
		 * from the top of the data stack and dereferences it.
		 *
		 * This is done when the module was automatically loaded by a ModReq; as
		 * the result of this operation is that of saving the module in the module space,
		 * there isn't any need to keep the module around in any other place.
		 */
		PStep* m_stepDisposeLoad;

		/** Stores a mantra dynamically found on a just loaded module in its place.
		 *
		 * During deserialization of mantras, it might be found that they depend
		 * on foreign mantras coming from modules that are not yet loaded.
		 *
		 * This step is used during the process of loading the module where the
		 * required mantra is stored. Once the module load is complete, this step
		 * searches for the required mantra in the newly loaded module, and if found,
		 * it puts it on top of the data stack.
		 */
		PStep* m_stepSaveDynMantra;

		/** Step called to pop the result of a main function.
		 *
		 * This step is pushed right before invoking a non-main module, and has two modes:
		 * with seqId == 1, will push the process result in the stack, with seqId == 0 will
		 * pop an item from the stack and set it as process result.
		 *
		 * With this operations, the VM is able to save the proper module result as result
		 * of a process created specifically to run a module main function.
		 *
		 */
		PStep* m_stepSetProcResult;

		/** Step used to invoke the main function of modules.
		 *
		 * The init step of modules can involve executing several init() methods
		 * of singleton objects.
		 *
		 * This must be performed prior invoking the main() function on that module.
		 *
		 * To achieve this, a step invoking the main function is first pushed, then
		 * all the required init operations are pushed. Once all the init have been
		 * completed, this step punches in and invokes the main function of the module.
		 */
		PStep* m_stepCallMain;

		// internal (not exported) forward class declaration.
		class PStepManagedLoadedModule;
		friend class PStepManagedLoadedModule;

		// internal (not exported) forward class declaration.
		class PStepResolveModReq;
		friend class PStepResolveModReq;

		// internal (not exported) forward class declaration.
		class PStepStoreModReq;
		// internal (not exported) forward class declaration.
		class PStepSetupModule;
		// internal (not exported) forward class declaration.
		class PStepCompleteLoad;
		// internal (not exported) forward class declaration.
		class PStepDisposeLoad;

		class PStepSaveDynMantra;
		class PStepSetProcResult;
		class PStepCallMain;



		FALCON_REFERENCECOUNT_DECLARE_INCDEC( ModSpace );
};

}

#endif

/* end of modspace.h */

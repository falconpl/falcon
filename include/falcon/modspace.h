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

namespace Falcon {

class VMContext;
class Module;
class Error;
class Symbol;
class ModLoader;
class ImportDef;
class Mantra;
class Process;
class Function;
class Variable;
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

   void resolveDeps( VMContext* ctx, Module* mod );

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
   Process* loadModule( const String& name, bool isUri, bool asLoad, bool isMain = false);

   void loadModuleInProcess( const String& name, bool isUri, bool asLoad, bool isMain = false);
   void loadModuleInProcess( Process* prc, const String& name, bool isUri, bool asLoad, bool isMain = false);

   void loadModuleInContext( const String& name, bool isUri, bool asLoad, bool isMain, VMContext* tgtContext );

   //===================================================================
   // Service functions
   //===================================================================
   
   void findDynamicMantra(
      VMContext* ctx,
      const String& moduleUri, 
      const String& moduleName, 
      const String& className );
   
      
   void gcMark( uint32 mark );
   
   uint32 currentMark() const { return m_lastGCMark; }
    
      
   /** Get the space in which this group resides. */
   ModSpace* parent() const { return m_parent; }
   
   
   /** Finds a module stored in this space by name.
    \param name The name of the module.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space,  or in the parent space(s) if there are parents.
    */
   inline Module* findByName( const String& name ) const {
      return findModule( name, false );
   }
   
   
   /** Finds a module stored in this space by its URI.
    \param name The URI of the module.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space,  or in the parent space(s) if there are parents.
    */
   inline Module* findByURI( const String& uri ) const { 
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
    */
   Error* exportSymbol( Module* source, const String& name, const Variable& var );

   /** Finds a value that has been generally exported via the load/export constructs.
    Finds a globally exported symbol.
    */
   Item* findExportedValue( const String& symName, Module*& declarer );
   
   /** Finds a value that might be generally exported or imported by a module.
    \param asker The module that is asking for the given symbol.
    \param symName The name of the symbol that is being searched (as remotely known).
    \param decalrer a place where to store the module that declared the symbol, if found.
    \return A valid symbol or 0 if the symbol is not found.
    
    This method finds a value that might be coming either from the global namespace
    generated in this ModSpace via export/load directives, or a generally imported symbol
    from any of the modules that were declared as general providers via import/from by
    the module that is searching for that symbol.
    
    The generic load/export search is extended to the parent ModSpaces, if 
    there is some parent.
    */
   Item* findExportedOrGeneralValue( Module* asker, const String& symName, Module*& declarer );

   inline Item* findExportedOrGeneralValue( Module* asker, const String& symName )
   {
      Module* declarer;
      return findExportedOrGeneralValue( asker, symName, declarer ); 
   }
   
   /** Finds a globally exported symbol.
    */
   inline Item* findExportedValue( const String& symName )
   {
      Module* declarer;
      return findExportedValue( symName, declarer );
   }
   
   
   /** Gets the module loader associated with this virtual machine. */
   ModLoader* modLoader() const { return m_loader; }

   /** Virtual machine associated with this space. */
   Process* process() const { return m_process; }

   /**
    * Returns the engine-wide visible instance of ClassModSpace;
    */
   static Class* handler();

private:      
   class Private;
   ModSpace::Private* _p;
   friend class Private;

   Process* m_process;
   ModSpace* m_parent;
   uint32 m_lastGCMark;
   Function* m_loaderFunc;
   
   ModLoader* m_loader;
   
   virtual ~ModSpace();
   void exportFromModule( Module* mod, Error*& link_errors );
   
   void importInModule(Module* mod, Error*& link_errors);
   void importInNS(Module* mod );
   void addLinkError( Error*& top, Error* newError );
   
   void importSpecificDep( Module* asker, void* def, Error*& link_errors);
   void importGeneralDep( Module* asker, void* def, Error*& link_errors);


   void retreiveDynamicModule(
      VMContext* ctx,
      const String& moduleUri, 
      const String& moduleName);


   class PStepLoader: public PStep
   {
   public:
      PStepLoader( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepLoader() {};
      virtual void describeTo( String& str, int ) const { str = "PStepLoader"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepLoader m_stepLoader;


   class PStepResolver: public PStep
   {
   public:
      PStepResolver( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepResolver() {};
      virtual void describeTo( String& str, int ) const { str = "PStepResolver"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepResolver m_stepResolver;


   class PStepDynModule: public PStep
   {
   public:
      PStepDynModule( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepDynModule() {};
      virtual void describeTo( String& str, int ) const { str = "PStepDynModule"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepDynModule m_stepDynModule;

   class PStepDynMantra: public PStep
   {
   public:
   PStepDynMantra( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepDynMantra() {};
      virtual void describeTo( String& str, int ) const { str = "PStepDynMantra"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepDynMantra m_stepDynMantra;

   class PStepExecMain: public PStep
   {
   public:
      PStepExecMain( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepExecMain() {};
      virtual void describeTo( String& str, int ) const { str = "PStepExecMain"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepExecMain m_stepExecMain;

   class PStepStartLoad: public PStep
   {
   public:
      PStepStartLoad( ModSpace* owner ): m_owner( owner ) {
         apply = apply_;
      }
      virtual ~PStepStartLoad() {};
      virtual void describeTo( String& str, int ) const { str = "PStepStartLoad"; }

   private:
      static void apply_( const PStep* self, VMContext* ctx );
      ModSpace* m_owner;
   };
   PStepStartLoad m_startLoadStep;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC(ModSpace);
};

}

#endif

/* end of modspace.h */

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
#include <falcon/symbolmap.h>

namespace Falcon {

class VMContext;
class Module;
class Error;
class Symbol;
class ModLoader;
class ImportDef;
class Mantra;

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
    \param ctx The VM Context used in this 
    \param parent A parent module space where to look for already loaded modules
    or exported symbols.
    */
   ModSpace( VMContext* ctx, ModSpace* parent = 0 );
   virtual ~ModSpace();
   
   /** Adds a new module to the module space. 
    \param module The module to be added.
    \param bExport If true, honor the export requests of the module.
    \return true if added, false otherwise.
    
    If the same identical module is present (by address), the method returns
    false. As multiple modules having the same logical name should not exist in
    the same space, the incoming module will automatically receive a different
    unique name (schemed after "modname-X", where X is an incremental number).
    If this is not desirable, check the result of getModuleByName() method
    before trying to add the module.
    
    \note This is the first step of creating a module space. It's optional,
    and should be only used if the module is known not to have any
    dependency.
    */
   bool add( Module* mod, bool bExport = false, bool bOnw = true );
   
   /** Adds the given module and recursively tries to load the dependencies.
    \param mod The Module that must be added to the space.
    \param bExport whether the space should honor the export request of the
            given module or not.
    \throw IoError on error finding or loading the required modules.   
    
    This methods calls add() on the given module and then resolveDeps to
    resolve all the dependencies of the added module.
    
    \note In case of error, the space should not be considered safe and
    be destroyed as soon as possible.
    
    \note This is the second step of creating the module space. The method
    repeatedly calls add() to store in the added modules in the module space.
    
    */
   void resolve( Module* mod, bool bExport = false, bool bOwn = true );
   
   
   /** Resolves an import request.
    
    \param ml A mod loader that is used to load 
    \param def The definition containing the module and the symbols to be resolved.
    \param requester A module where the resolved symbols are stored. Can be null.
    \throw IoError on error finding or loading the required modules.
    \throw A link error with suberrors in case of errors during the linkage.
    
    This method checks or eventually performs all the linkages that are specified
    by the given import definition. If a target module is given, its onImportResolved
    is repeatedly called each time a symbol is resolved, and if the import directive
    contains a namespace import requests (e.g. import * from...) then the onImportAll
    is repeatedly called. In both cases, the name provided is the one that the
    symbol should have in the requester module (accordingly with the import
    definition).
    
    An empty requester can be used to pre-load the modules and check for correctness
    of the import directive before actually applying it on a requester at a later
    time.
    */
   void resolveImportDef( ImportDef* def, Module* requester = 0 );
   
   
   /** Just resolves the dependencies of a module that was separately added.
    \param mod The Module that must be added to the space.
    \param bExport whether the space should honor the export request of the
            given module or not.
    \throw IoError on error finding or loading the required modules.   
    
    This method is usually called by resolve().
    
    */
   void resolveDeps( Module* mod );
   
   /** Links newly added modules.
    \return A link error containing all the problems in this link
    step (as sub errors), or 0 if the link process was ok.
    
    This method resolves all the link export and import requests pending from
    the previous add() and resolve steps.
    
    In case of error, an Error containing one or more sub-errors is returned.
    In that case, the space should be considered invalid and destroyed when
    possible.
    
    Calling link() without a prior add() or reslove() method call has no
    effect.
    
    \note This is the third step of creating the module space.     
    */
   Error* link();   
    
    /** Readies the call to initialize the modules in the VM.
     \param ctx A VMContext The context where the initialization shall take place.
     \return True if there was some initialization function to be added to the
     context, false if there isn't any initialization procedure scheduled.
     
     This method stores the calls to the __main__ functions and object initialization
     methods that each module require to be performed in the given context, 
     in proper load order (last module added will have its initialization
     sequence called first). 
     
     A subsequent execution of the VM containing the given context will invoke
     the intialization procedures in proper order.
     
     Calling this method clears the list of initialization procedures scheduled
     for execution, so multiple calls to this method have no effect.
     
     \note This is the fourth step of creating the module space.     
    */
   bool readyVM( VMContext* ctx );
    
   //===================================================================
   // Service functions
   //===================================================================
   
   Mantra* findDynamicMantra( 
      const String& moduleUri, 
      const String& moduleName, 
      const String& className, 
      bool &addedMod );
   
   
   /** Returns a previously stored module by name.
    \param name The logical name of the module to be searched.
    \return 0 if not found, a valid module if found.
    */
   Module* getModuleByName( const String& name);
      
   void gcMark( uint32 mark );
   
   uint32 lastGCMark() const { return m_lastGCMark; }
   
   /** Returns all the generated error as a single composed error. 
    \return a LinkError of type e_link_error containing all the suberrors, 
    or 0 if there was no error to be returned.
    
    Once called, the previusly recorded errors are cleared.
    */
   Error* makeError() const;
    
      
   /** Get the space in which this group resides. */
   ModSpace* parent() const { return m_parent; }
   
   /** Finds a module stored in this space by name.
    \param name The name of the module.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space, 
    or in the parent space(s) if there are parents.    
    */
   inline Module* findByName( const String& name ) const { 
         bool bDummy; 
         return findByName(name, bDummy);
   }
   
   /** Finds a module stored in this space by name.
    \param name The name of the module.
    \param bExport will be set to true if the module is scheduled to export its variables.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space, 
    or in the parent space(s) if there are parents.    
    */
   Module* findByName( const String& name, bool& bExport ) const;
   
   
   /** Finds a module stored in this space by its URI.
    \param name The URI of the module.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space, 
    or in the parent space(s) if there are parents.    
    */
   inline Module* findByURI( const String& uri ) const { 
         bool bDummy; 
         return findByURI(uri, bDummy);
   }
   
   /** Finds a module stored in this space by URI.
    \param uri The URI of the module.
    \param bExport will be set to true if the module is scheduled to export its variables.
    \return A module if found, 0 otherwise.
    
    This method searches a module in this space, 
    or in the parent space(s) if there are parents.    
    */
   Module* findByURI( const String& uri, bool& bExport ) const;

   /** Exports a single symbol on the module space. 
    */
   Error* exportSymbol( Module* mod, Symbol* sym );

   /** Finds a symbol that has been generally exported via the load/export constructs.
    Finds a globally exported symbol.
    */
   Symbol* findExportedSymbol( const String& symName, Module*& declarer );
   
   /** Finds a symbol that might be generally exported or imported by a module.
    \param asker The module that is asking for the given symbol.
    \param symName The name of the symbol that is being searched (as remotely known).
    \param decalrer a place where to store the module that declared the symbol, if found.
    \return A valid symbol or 0 if the symbol is not found.
    
    This method finds a symbol that might be coming either from the global namespace
    generated in this ModSpace via export/load directives, or a generally imported symbol
    from any of the modules that were declared as general providers via import/from by
    the module that is searching for that symbol.
    
    The generic load/export search is extended to the parent ModSpaces, if 
    there is some parent.
    */
   Symbol* findExportedOrGeneralSymbol( Module* asker, const String& symName, Module*& declarer );

   inline Symbol* findExportedOrGeneralSymbol( Module* asker, const String& symName )
   {
      Module* declarer;
      return findExportedOrGeneralSymbol( asker, symName, declarer ); 
   }
   
   /** Finds a globally exported symbol.
    */
   inline Symbol* findExportedSymbol( const String& symName )
   {
      Module* declarer;
      return findExportedSymbol( symName, declarer );
   }
   
   /** Links the unlinked imports newly found in a single module.
    \param mod Module.
    */
   Error* linkModuleImports( Module* mod );
   
   /** Gets the module loader associated with this virtual machine. */
   ModLoader* modLoader() const { return m_loader; }
   
   /** Gets the active context bound with this module loader. */
   VMContext* context() const { return m_ctx; }
   
private:      
   class Private;
   ModSpace::Private* _p;
   friend class Private;
   
   class ModuleData;
   
   VMContext* m_ctx;
   ModSpace* m_parent;
   uint32 m_lastGCMark;
   
   ModLoader* m_loader;
   
   ModuleData* findModuleData( const String& name, bool isUri ) const;   
   void exportFromModule( Module* mod, Error*& link_errors );
   
   void linkImports(Module* mod, Error*& link_errors);
   void linkNSImports(Module* mod );
   void addLinkError( Error*& top, Error* newError );
   
   void linkSpecificDep( Module* asker, void* def, Error*& link_errors);
   void linkGeneralDep( Module* asker, void* def, Error*& link_errors);
   
   Module* retreiveDynamicModule( 
      const String& moduleUri, 
      const String& moduleName,  
      bool &addedMod );

};

}

#endif

/* end of modspace.h */
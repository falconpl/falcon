/*
   FALCON - The Falcon Programming Language.
   FILE: module.h

   Falcon code unit
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Feb 2011 14:37:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_MODULE_H
#define FALCON_MODULE_H

#include <falcon/atomic.h>
#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/enumerator.h>
#include <falcon/modmap.h>
#include <falcon/loadmode.h>
#include <falcon/mantra.h>
#include <falcon/class.h>
#include <falcon/function.h>
#include <falcon/module.h>
#include <falcon/globalsmap.h>
#include <falcon/attribute.h>
#include <falcon/atomic.h>
#include <falcon/service.h>

#define DEFALUT_FALCON_MODULE_INIT falcon_module_init
#define DEFALUT_FALCON_MODULE_INIT_NAME "falcon_module_init"

#define FALCON_MODULE_DECL \
   FALCON_MODULE_TYPE DEFALUT_FALCON_MODULE_INIT()

namespace Falcon {

class Symbol;
class Item;
class ModSpace;
class ModLoader;
class FalconClass;
class DynUnloader;
class Requirement;
class ImportDef;
class ModRequest;
class ClassModule;
class TextWriter;
class DynLibrary;

/** Standard Falcon Execution unit and library.

 Falcon modules are used to transiently store code and data that can then
 be executed by a virtual machine (usually, a single virtual machine, and just once).

 The contents of modules is dynamic in the sense that it might be altered at
 any stage of their lifetime.

 Modules can be linked in a virtual machine statically or dynamically.
 
 Static modules have a lifetime that is meant to be longer than that of the
 virtual machine they are attached to. They are injected into the virtual machine
 externally and cannot be explicitly referenced or unloaded by the owning VM.

 Dynamic modules can be loaded and unloaded multiple times during the lifetime
 of their own VM.
 
 The main difference between the two type of modules is that dynamic modules need
 to create entities that might survive their own lifetime, and that need to be
 accounted separately for garbage reclaim.
 
 The advantage of statically link a module is that all the items it declares
 are outside the scope of garbage collection. This has two effects: first,
 a program declaring linking a module statically doesn't need
 any special care about preventing that module to be destroyed by the garbage
 collector, and stays in control of its lifespan. Second, the garbage
 collector is not required to perform useless checks on the items declared by
 the module (mainly functions and classes), that will be considered always
 valid.

 The kind of a module is not stored in the module structure itself. Instead,
 it's the link process that determines if a module should be considered static
 or dynamic. Notice that the link process @b modifies the contents of the module
 and it's not possible to link the same module twice (but it is possible to clone
 a "model" module dispatching its clone for linkage any number of time).

 \note Function and Class have support to gc-mark the modueles they come
 from, so that dynamic modules that might be unloaded by the virtual machine
 at system process level stay alive as long as there is at least one function or
 class getting CPU code directly from their memory space. SynFunction class owns 
 its syntactic tree and reference global variables, so they don't need to back-mark 
 their module -- however, they do because of meta-information held in the module
 (module logical name, URI, attributes etc).
 
 \note Modules whose source is uncertain should be always deleted through the
 unload() method. See the method description for more information.
 */
class FALCON_DYN_CLASS Module {
public:

   Module();
   
   /** Creates an internal module.
    \param name The symbolic name of this module.
    \param isNative Native modules don't get automatically serialized.
    */
   Module( const String& name, bool isNative = true );

   /** Creates an external module.
    \param name The symbolic name of this module.
    \param uri The uri from where this module was created.
    \param isNative Native modules don't get automatically serialized.
    */
   Module( const String& name, const String& uri, bool isNative = true );

   /** Logical name of the module. */
   const String& name() const { return m_name; }

   /** Physical world location of the module. */
   const String& uri() const {return m_uri;}

   /** Creates a new global variable in the module (or promotes an extern).
    * \param name The name of the global to be added.
    * \param value The value of the new global.
    * \return The data of the added global, or 0 if the global name is not available.
    *
    */
   GlobalsMap::Data* addGlobal( const String& name, const Item& value, bool bExport = true );

   /** Creates a new global variable in the module (or promotes an extern).
    * \param sym The symbol of the global to be added.
    * \param value The value of the new global.
    * \return The data of the added global, or 0 if the global name is not available.
    *
    */
   GlobalsMap::Data* addGlobal( const Symbol* sym, const Item& value, bool bExport = true );


   /** Adds a global mantra, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A variable containing the global ID or 0 if the variable
             was already declared.

      Other than being added to the static values and to the global
      variables table, the function will be added also to the module mantra table.

      This means that after calling this method, the name of the mantra is also known as
      a global, possibly exported variable.
    */
   bool addMantra( Mantra* f, bool bExport = true );
   
   /** Adds a class and the required structures to make it initializable at startup.
    *
    * Creates also an object entity to be initialized at module startup.
    * The name of the class MUST start with "%", that marks an initializable
    * class.
    *
    * \note this is for internal usage. Use addObject instead.
    *
    */
   bool addInitClass( Class* cls, bool bExport = true );

   /** Creates a singleton object that is initialized at startup
    *
    * The class is to be exclusively used as a base for this object.
    * If it doesn't start with a "%" character, it is automatically
    * added.
    *
    */
   bool addObject( Class* cls, bool bExport = true );

   /**
    * Returns the count of classes with an init-time instance to be filled.
    */
   int32 getInitCount() const ;

   /**
    * Returns the nth class with an init-time instance to be filled.
    */
   Class* getInitClass( int32 val ) const;


   /** Adds a constant to the module.
    * @param name The name of the constant.
    * @param value The value of the constant.
    * @param bExport True to make the constant exported.
    * \return true if the constant can be defined, false if it was already defined.
    *
    * Creates a global variable which will be initialized to the required
    * value at load time.
    *
    * \note, contrarly to addGlobal, addConstant doesn't promote external values.
    */
   bool addConstant( const String& name, const Item& value, bool bExport = true );

   /** Adds an anonymous mantra.
    \param f The mantra to be added

    The name of the mantra will be modified so that it is unique in case
    it is already present in the module.
    
    The mantra will not be exported, and there won't be any variable created
    for this mantra.
    */
   void addAnonMantra( Mantra* f );


   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A variable containing the global ID or 0 if the variable
             was already declared.

     This is a candy grammar for creating an ExtFunc entry and adding it
     as a Mantra to the module.

      Other than being added to the static values and to the global
      variables table, the function will be added also to the module mantra table.

      This means that after calling this method, the given name is also known as
      a global, possibly exported variable.
    */
   Function* addFunction( const String& name, ext_func_t f, bool bExport = true );


   /** Finds a function.
    \param name The function name to be searched.
    \return A global function or 0 if not found.

    If the given name is present as a global function in the current module.
    */
   Mantra* getMantra( const String& name, Mantra::t_category cat = Mantra::e_c_none ) const;


   /** Enumerator receiving mantras in this module. */
   typedef Enumerator<Mantra> MantraEnumerator;

   /** Enumerate all functions and classes in this module.
    */
   void enumerateMantras( MantraEnumerator& rator ) const;


   /** Candy grammar to add exported functions. */
   Module& operator <<( Mantra* f )
   {
      addMantra( f );
      return *this;
   }


   /** Adds a generic import request.
    \param source The source path or logical module name.
    \param bIsUri If true, the source is an URI, otherwise is a logical module name.
    \return True if the module could be added or promoted, false if it was
            already imported as generic.
   
    Generic import requests (as in "import from ModuleName") inform the
    engine that unknown symbols declared in the module are to be searched 
    first in that module, and then in the global exported symbol space.
    
    */
   bool addGenericImport( const String& source, bool bIsUri );

   /** Mark (dynamic) modules for Garbage Collecting.
    \param mark the current GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */
   void gcMark( uint32 mark );

   /** Determines if a module can be reclaimed.
    \return last GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */

   uint32 currentMark() const { return m_lastGCMark; }

   /** Adds a request to load or import a module.
    *
    * \param name Logical name or URI of the target module.
    * \param isUri if true, the given name is to be intended as an URI, otherwise it's
    *        a logical module name.
    * \param isLoad if true, the module is loaded with load semantic (static export
    *        requests will be honored).
    *
    * When a module gets deserialized or anyhow loaded in the engine, it can ask
    * for a set of other modules to be loaded in the same module space before
    * the module can run.
    *
    * The engine will store the required modules, or retrieve the modules in the
    * same target module space if they were already loaded by something else,
    * and fill the appropriate ModRequest::module() field prior calling onLinkComplete()
    * on this module.
    *
    * If any of the required modules cannot be found, the engine stops the load process
    * with error, and the onLinkComplete() method of this module is never called. Notice that
    * processes can invoke functions that explicitly load modules in a sandbox process,
    * like in the case of the include() function, so this doesn't mean that the VM
    * will stop.
    *
    * \note There is no guarantee on the call order of the onLinkComplete() method of the
    * modules invoked in the modRequest. As onLinkComplete() in this module gets called,
    * the modules stored in the created ModRequest entries might or might not already
    * be initialized or even run. However, they will have been fully linked, and their
    * constructor and onLoad() methods (where static class/function definition are honored)
    * will have been called.
    *
    */

   ModRequest* addModRequest( const String& name, bool isUri, bool isLoad = false );

   /** Gets the count of external module dependencies in this module.
    * \return count of external module dependencies statically declared by this module.
    */
   length_t modRequestCount() const;

   /** Gets the nth external module dependency in this module.
    * \param id position of the module dependency.
    * \return 0 if id is out of range, a valid ModRequest* otherwise.
    */
   ModRequest* getModRequest(length_t id) const;

   /** Callback invoked by the engine as soon as the module is considered "live".
    *
    * This method is invoked when a module is inserted in a module space. This is the moment
    * in which the module is considered to be "alive" and ready to respond to engine requests.
    *
    * After the method returns, the static module request resolution starts; so, in this moment
    * ModRequest entries are still void.
    *
    * This is a good spot to add dynamic entries that couldn't be possibly added in the Module
    * subclass constructor.
    *
    * New ModRequest and ImportDef can be added during this step.
    *
    * \note At time of call, the modSpace() member is already filled-in and the module is already
    *       stored in the parent modSpace. Also, usingLoad() is correctly set.
    */
   virtual void onLoad();
   
   /** Called back by the engine when a required module is found and loaded.
    * \param mr The module request that has just been filled.
    *
    * This method is called back as soon as the ModRequest::module() member is filled
    * with a Module*. It is guaranteed that the module that is loaded has been correctly
    * initialized, and its onLoad() method has been called, but there is no other
    * guarantee on the lifecycle of the given module.
    *
    * For instance, if the module was already present in the parent module space,
    * it might have already been used for running code, while, if it was freshly loaded,
    * it might not yet have had its onLinkComplete() method called.
    *
    * \note The onLinkComplete() method is called as soon as all the ModRequest are resolved (and
    *  onModuleResolved is called for each one of them).
    */
   virtual void onModuleResolved( ModRequest* mr );

   /** Called back when a static import definition is resolved.
    * \param id The import definition that generated the symbol.
    * \param sym The symbol that was resolved.
    * \param value The value asociated with the symbol.
    *
    * This is called back when an explicit static import definition is
    * resolved by the module space.
    */
   virtual void onImportResolved( ImportDef* id, const Symbol* sym, Item* value );

   /** Callback invoked by the engine after the link process is complete.
    *
    * \param ctx The context where the link has been completed.
    *
    * This method is called when the static link process is considered complete.
    *
    * This happens when all the statically declared ModRequest have been honored (all
    * the required modules are loaded) and after all the statically declared external
    * symbols have been resolved, and prior any code of the module (including static object
    * initialization) is run.
    *
    * This is a good place where to search for symbols defined in foreign modules, or that should
    * be found in the module space, and cache them in the globals map and/or in local C++ members.
    *
    * \note The base version of this method does nothing.
    *
    * \note The method can throw an error if, for any reason, the initialization of the module
    * can't be performed.
    */
   virtual void onLinkComplete( VMContext* ctx );

   /** Invoked by the loader when all the init functions and the main function have been called.
    *
    * \note The loader may skip calling the main function if this is a main module, or if it is
    * explicitly required not to call the main function at load.
    */
   virtual void onStartupComplete( VMContext* ctx );

   /** Invoked when the module is removed from the modspace.
    *
    * This happens right before the module is removed.
    */
   virtual void onRemoved( ModSpace* ms );

   /** Perform live module initialization on a virtual machine.
    * \param VMContext The context where the module is invoked.
    *
    * This method prepares the execution of code declared within the module (if any)
    * in the target context.
    *
    * This is articulated in three steps:
    * - Initialization of falcon classes (if any).
    * - Initialization of attributes (if any).
    * - Initialization of singleton objects (if any).
    *
    * This steps may be executed immediately or pushed in the context as PSteps for
    * later execution; there is no guarantee about the execution time, but it is guaranteed
    * that they will be executed in the given order.
    *
    * Subclasses can extend this functionality. If they don't provide falcon classes, singleton
    * objects and a syntactic (PStep-tree based) main function, they can safely ignore the base method.
    * Also, the base method can be ignored if the subclass knows how to initialize its own FalconClass
    * instances.
    *
    * In all the other cases, the derived classes should invoke the base method immediately, and then
    * eventually add their own initialization code to the context, in order for that to be executed
    * first (the PSteps in the context are a LIFO stack: later pushed PSteps are executed first).
    *
    * \note The module should never invoke its own main function; this is a decision that
    *     is demanded to the module loader, either the default one in the ModSpace or a dynamic
    *     loader deserializing the module on its own.
    *
    * \note The caller may check if this method has left pending PSteps operations in the context
    *       by confronting the depth of the code stack prior and after invoking it.
    */
   virtual void startup( VMContext* ctx );

   /** Adds a standard import/from or load request.
    \param def An import definition.
    \param error An output error that is created in case of error.
    \return true if the import definition is compatible with the module,
    false if the symbols are already defined or if a load request was
    already issued for a newly loaded module.
    
    The method adds an import definition (corresponding to a "import" directive line
    in the source) to the current module.

    Several checks are performed in order to ensure that the import request is coherent
    with the module. In particular:
    - Output (locally visible) symbols declared in the import definition must not be
      already declared.
    - The module declared in the import directive (if any) must not have been
      already declared as loaded.

    In case of error, the method returns false and a consistent newly allocated error
    will be placed in the \b error parameter. The caller might add it to
    a list of errors or raise it immediately.

    \note Multiple if the import directive generates multiple errors, they are stored
    as sub-errors of a LinkError.
    */
   bool addImport( ImportDef* def, Error*& error, int32 line = 0 );
   
   /** Adds an implicitly imported symbol.
    Proxy to the version using a Symbol as parameter.
    */
   bool addImplicitImport( const String& name, int32 line = 0 );

   /** Adds an implicitly imported symbol.
    \param name The unknown symbol that should be resolved during link time.
    \param line The line where the symbol is defined.
    \return true if the symbol is new, false if it was already defined.

    \note If the method returns false, the caller should raise a LinkError for
    doubly defined symbol.
    */
   bool addImplicitImport( const Symbol* sym, int32 line = 0 );
   
   /** Shortcut to create a load ImportDef only if load is valid.
    \param name The name or URI of the module.
    \param bIsUri if name is URI.
    \param Error a reference to a Error* that will be filled in case of error.
    \return true if the load request can be added, false if the module is
       already declared as required for load.
    
    */
   bool addLoad( const String& name, bool bIsUri, Error*& error, int32 line = 0 );

   /** Callback that is called when a symbol import request is satisfied.
    \param requester The module from which the request was issued.
    \param definer The module where the symbol has been found.
    \param sym The resolved symbol.
    */
   
   typedef Error* (*t_func_import_req)( const Module* sourceModule, const String& sourceName, Module* targetModule, const Item& value, const Variable* targetVar );

   /** Reads the exportAll flag.
    \return True if this module wants to export all the non-private symbols.
    */
   bool exportAll() const;
   
   /** Sets the exportAll flag.
    \param e set to true if this module wants to export all the non-private symbols.
    */   
   void exportAll( bool e );
   
   /** Returns the module group associated with this module.    
    \return The module group in which this module is stored, or 0 if this
            is a static module living in the global module space.
    
    The module space might be created either by the module itself,
    and used for private load, or be assigned by the loading process
    by other modules or by the loading module space.
    */
   ModSpace* modSpace() const { return m_modSpace; }
   
   void modSpace( ModSpace* md ) { m_modSpace = md; }
   
   
   bool isMain() const { return m_bMain; }
   
   void setMain( bool isMain = true ) { m_bMain = isMain; } 

   /** Check if we can use an existing module to resolve this requirement.
    
    We can use a module that has been already loaded if:
    - It's local (modgrouop), it has been imported privately and need to import it privately here.
    - It's local, and is imported publically or loaded (if not loaded, might get promoted).
    - It's global (modspace), and it's \b not imported privately here.
    
    Otherwise, a new load is needed (even if the module already exists somewhere.
    */
   Module* linkExistingModule( const String& name, bool bIsUri, t_loadMode imode );
   
   Function* getMainFunction();
   void setMainFunction( Function* mf );
   
   bool isNative() const { return m_bNative; }
   
   /**
    * Sets the load mode of this module.
    *
    * Modules can be imported in a module space with the load or import directives.
    *
    * The load directive requests the engine to honor export requests from the module,
    * so that all the loaded modules form a single body of application.
    *
    * Import directive makes the symbols defined in the module locally visible to the
    * importer, on its explicit request. It is mainly meant to use a foreign module
    * as a library of mantras to be used at will.
    *
    * This method indicates if the module is to be intended as loaded or imported in
    * a module space.
    *
    * @note A module cannot be used in multiple module spaces. Once assigned, it
    * must be exclusively used there, so the even if this setting depends on the
    * way the module is included in the module space, it can be safely stored as a module
    * characteristic.
    */
   void usingLoad( bool m ) { m_bLoad = m; }

   /**
    * Indicates current load/import setting for this module.
    * @see void usingLoad(bool)
    */
   bool usingLoad() const { return m_bLoad; }

   Class* getClass( const String& name ) const { return 
      static_cast<Class*>( getMantra(name, Mantra::e_c_class) ); }
      
   Function* getFunction( const String& name ) const { return 
      static_cast<Function*>( getMantra(name, Mantra::e_c_function) ); }

   /** Count of import definitions (dependencies). */
   uint32 importCount() const;

   /** Get the nth import definition (dependency).
    * \param n Definition number in range 0 <= n < depsCount().
    * \return ImportDef for that number.
    */
   ImportDef* getImport( uint32 n ) const;

   /** Access to the global variables map of this module. */
   GlobalsMap& globals() { return m_globals; }

   /** Access to the global variables map of this module. */
   const GlobalsMap& globals() const { return m_globals; }

   /** Resolves a symbol that is requested as a global entity by module users.
    * \param Symbol the symbol to be found.
    * \return 0 if the symbol is not found in the engine, otherwise a valid
    *    item known as the given symbol in this module.
    *
    * The symbol is searched looking at this module and in its dependencies in the
    * following order:
    *
    * - local search (through resloveLocally() method).
    * - search in modules providing symbols through import declarations.
    * - search in the holder module space (if any): this resolves in searching explicit
    *   exports from other modules imported in the module space via "load" directive.
    * - search in the engine built-ins.
    *
    * If the symbol is not found, 0 is returned.
    *
    * \note When a symbol is resolved, it is locally cached in the globals map. This means
    * that the resolution of unknown symbols is performed once only (it is not possible
    * to provide different item* at a later invocation),
    *
    * \note As globals get serialized when the module is stored, serializing a module after it has been linked
    * or run will have the effect of storing the resolved globals with it; this means that deserializing
    * that module afterwards will restore the dependencies automatically. In other words, the dependencies
    * that were recorded after looking at the import definition table will now be statically re-created at
    * de-serialization and won't be looked up anymore. Notice that the engine never serializes a module
    * after link or run; this consideration is relevant just to user-created modules that are explicitly
    * saved on a stream after being run via a Storer object.
    *
    */
   Item* resolve( const Symbol* sym );

   /** Performs resolution of locally defined global symbols.
    * \param The symbol to be resolved.
    * \return A value associated with this symbol, or 0 if the symbol is not locally known.
    *
    * This method returns a the value associated with a symbol as locally
    * known by this module.
    *
    * The default version of this method returns a symbol in the globals map. Subclasses
    * of the Module class might dynamically provide values as they are requested by importers.
    */
   virtual Item* resolveLocally( const Symbol* sym );
   Item* resolveLocally( const String& symName );

   /** Performs resolution of non-locally defined global symbols.
    * \param The symbol to be resolved.
    * \return A value associated with this symbol, or 0 if the symbol is not locally known.
    *
    * This method returns a the value associated with a symbol as ignoring local definitions
    * of the given symbol. This performs a search in the imported modules and exports in
    * the module space, plus a search in the engine.
    *
    * This method comes useful when the imported symbol is already defined in the module,
    * (i.e. as implicit import) and the actual value is to be found in one of the related
    * modules.
    */
   Item* resolveGlobally( const Symbol* sym );
   Item* resolveGlobally( const String& name );

   /** Resolves a symbol in this module.
    * \param name the name of the symbol to be resolved.
    * \return  0 if the symbol is not found in the engine, otherwise a valid
    *    item known as the given symbol in this module.
    *
    *  This version of resolve( const Symbol* sym ) can be used if a symbol is to be searched
    *  by name and the pointer to the globally known symbol entry in the engine is not
    *  readily available.
    */
   Item* resolve( const String& symName );

   /** Resolve all the imported values.
    * \param error On failure, will hold a LinkError with relevant information.
    * \return true on success, false on failure.
    */
   bool resolveImports( Error*& error );

   /** Returns the attributes for this entity. */
   const AttributeMap& attributes() const { return m_attributes; }

   /** Returns the attributes for this entity. */
   AttributeMap& attributes() { return m_attributes; }

   /**
    * Adds an international string to the current module.
    *
    * This strings can then be separately enumerated and saved.
    */
   void addIString( const String& iString );

   typedef Enumerator<String> IStringEnumerator;

   /** Enumerate all the international strings declared in this module.
    * @param cb An enumerator receiving the strings one at a time.
    */
   void enumerateIStrings( IStringEnumerator& cb ) const;

   /** Returns the number of international strings that can be enumerated in this module.
    *
    * This is useful in case of serialization, to know the number of international strings in advance.
    */
   uint32 countIStrings() const;

   void incref() const { atomicInc(m_refcount); }
   void decref();

   /**
    * Generates a full source representation of the module.
    */
   void render( TextWriter* tw, int32 depth );

   /** Returns an object natively exported in the module dynamic file.
    * \param name The name of the data/object/function to be searched.
    * \return a native data/object/function pointer or 0 if the object is not found --
    *    or if the module is not native.
    *
    * The isNative() method might be used to know if the module has a
    * native dynamic library backup or not.
    */
   void* getNativeEntity( const String& name ) const;

   /** Creates a service with the given name.
    *
    * A service is dynamically linked library interface to
    * a functionality offered by a module. It's meant to be implemented
    * as a full virtual-pointer method vector class, so that it can
    * be used by third party modules via the virtual methods through
    * a DLL interface without any need for direct linkage.
    *
    * In other words, a user of a service should be just able to load
    * the module and then use the header file of the service class to
    * access its functions, without a C++/system linkage getting in the way.
    *
    * This methods is meant to be overloaded by modules to create an
    * usable instance of a service, in case they want to publish it
    * to third party users that don't want to use the Class interface.
    */
   virtual Service* createService( const String& name );

   /** Compute the logical name of a module trimming 'self' and initial '.' in module specification.
    * \param Name the name of the module as required by a loader.
    * \param logicalName where to place the final name that should be assigned to the module once loaded.
    * \param loaderName If there is a module loading the one which name is to be computed,
    *        this is the name of the loader module. Else, it can be an empty string.
    * \return A reference to logicalName.
    *
    * A module specification can be absolute or relative with respect to the loader module.
    * The 'self' keyword is used to let the loaded module to inherit the loader module full name,
    * while an initial '.' is used to declare that the module is to be named as sibling of the
    * loader module.
    *
    * For instance, if the following code is found in the module "top.middle:"
    * @code
    * //In module "top.middle"
    *
    * load another       // 1
    * load .sister       // 2
    * load self.child    // 3
    * @endcode
    *
    * Then, the final name of the modules shall be the following:
    * # "another": and it will be searched in in any topmost directory in the search path
    * # "top.sister": and it will be searched in the "top/" subdirectory below the topmost search path entries.
    * # "top.middle.child": and it will be searched in "top/middle/" subdirectory below paths.
    *
    */
   static String& computeLogicalName( const String& name, String& logicalName, const String& loaderName = "" );

protected:
   /** Invoked when refcount hits 0.
    *  This will invoke the unload() method if not previously invoked.
    */
   virtual ~Module();

private:
   class Private;
   Module::Private* _p;
   
   GlobalsMap m_globals;

   ModSpace* m_modSpace;

   String m_name;
   String m_uri;
   uint32 m_lastGCMark;
   bool m_bExportAll;
   DynLibrary* m_dynlib;
   bool m_bMain;
      
   int m_anonMantras;
   Function* m_mainFunc;
   bool m_bNative;
   bool m_bLoad;
   mutable atomic_int m_refcount;

   AttributeMap m_attributes;

   friend class Private;   
   friend class DynLoader;
   friend class FAMLoader;
   friend class ModSpace;
   friend class ClassModule;
   
   void name( const String& v ) { m_name = v; }
   void uri( const String& v ) { m_uri = v; }
   void setDynUnloader( DynLibrary* ul ) { m_dynlib = ul; }
};

}

#endif	/* MODULE_H */

/* end of module.h */

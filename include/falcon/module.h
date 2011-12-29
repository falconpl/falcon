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

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/function.h>
#include <falcon/enumerator.h>
#include <falcon/refcounter.h>
#include <falcon/modmap.h>
#include <falcon/loadmode.h>

#define DEFALUT_FALCON_MODULE_INIT falcon_module_init
#define DEFALUT_FALCON_MODULE_INIT_NAME "falcon_module_init"

#define FALCON_MODULE_DECL \
   FALCON_MODULE_TYPE DEFALUT_FALCON_MODULE_INIT()

namespace Falcon {

class Symbol;
class Item;
class Class;
class Inheritance;
class ModSpace;
class ModLoader;
class FalconClass;
class DynUnloader;
class Requirement;
class ImportDef;

/** Standard Falcon Execution unit and library.

 Falcon modules are used to transiently store code and data that can then
 be executed by a virtual machine (usually, a single virtual machine, and just once).

 The contents of modules is dinamyc in the sense that it might be altered at
 any stage of their lifetime.

 Modules can be linked in a virtual machine statically or dynamically.
 
 Static modules have a lifetime that is meant to be longer than that of the
 virtual machine they are attached to. They are injected into the virtual machine
 externally and cannot be explicitly referenced or unloaded by the owning VM.

 Dynamic modules can be loaded and unloaded multiple times during the lifetime
 of their own VM.
 
 The main differece between the two type of modules is that dynamic modules need
 to create entities that might surivive their own lifetime, and that need to be
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

   /** Creates an internal module.
    \param name The symbolic name of this module..
    */
   Module( const String& name );

   /** Creates an external module.
    \param name The symbolic name of this module.
    \param uri The uri from where this module was created.
    */
   Module( const String& name, const String& uri );

   virtual ~Module();

   /** Logical name of the module. */
   const String& name() const { return m_name; }

   /** Physical world location of the module. */
   const String& uri() const {return m_uri;}


   /** Add static data that must be removed when the module is destroyed.
    \param cls Data class.
    \param data Data entity.
    
    Strings, anonymous functions and classes and so on are to be destroyed
    when the module is gone.
    
    Static data should not be added to dynamic module (where data must be
    able to take care of itself.
    
    The source compiler will call addAnonFunciton or addAnonClass exactly
    after having compiled a valid anonymous function or class. To prevent
    leaks in case of compilation errors, they are first dispatched to this
    method. The module will remove the anonymous elements from the static
    data list when they are found at the back of the list.
    */
   Item* addStaticData( Class* cls, void* data );

   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A Symbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   Symbol* addFunction( Function* f, bool bExport = true );

   /** Adds an anonymous function.
    \param f The function to be added

    The name of the function will be modified so that it is unique in case
    it is already present in the module.
    */
   void addAnonFunction( Function* f );

   /** Storing it on an already defined symbol.
    \param sym The global symbol that is already stored on this module.
    \param f The function to be added

    
    */
   void addFunction( Symbol* sym, Function* f );

   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.

    */
   Symbol* addFunction( const String& name, ext_func_t f, bool bExport = true );

   /** Adds a new class to this module.
    \note This method doesn't seek for existing waiting 
    */
   void addClass( Symbol* gsym, Class* fc, bool isObj );

   /** Adds a global class, possibly exportable.
    \param fc The class to be added
    \param isObj If true, there's a singleton instance bound to this class.
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   Symbol* addClass( Class* fc, bool isObj, bool bExport = true );

   /** Finds an already registered class.
    \return A global class or 0 if not found.

    If the given name is present as a global class in the current module.
    */
   Class* getClass( const String& name ) const;
   
   
   /** Adds an anonymous class.
    \param cls The class to be added

    The name of the class will be modified so that it is unique in case
    it is already present in the module.
    */
   void addAnonClass( Class* cls );

   /** Adds a global variable, possibly exportable.
    \param name The name of the symbol referencing the variable.
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            value is now stored, or 0 if the function name is already present.

    Creates a nil variable and references it to a global symbol.
    */
   Symbol* addVariable( const String& name, bool bExport = true );

   /** Adds a global variable, possibly exportable.
    \param name The name of the symbol referencing the variable.
    \param bExport if true, the returned symbol will be exported.
    \param value the value to be added.
    \return A GlobalSymbol holding a pointer to the global variable where the
            value is now stored, or 0 if the function name is already present.

    Creates an already valorized variable in the module global vector.
    \note The garbage collector may be running while performing this operation.
    If the data to be added is a garbageable deep data, be sure to allocate
    a garbage lock that can be released after the module has been linked in
    the virtual machine.
    */
   Symbol* addVariable( const String& name, const Item& value, bool bExport = true );

   /** Finds a global symbol by name.
    \param name The symbol name to be searched.
    \return A global symbol (either defined or undefined) or 0 if not found.
    
    If the given name is not present as a global symbol in the current module,
    an imported UnknownSymbol will be added.

    \note The returned symbol might be a GlobalSymbol or an UnknownSymbol.
    */
   Symbol* getGlobal( const String& name ) const;

   /** Finds a function.
    \param name The function name to be searched.
    \return A global function or 0 if not found.

    If the given name is present as a global function in the current module.
    */
   Function* getFunction( const String& name ) const;

   /** Enumerator receiving symbols in this module. */
   typedef Enumerator<Symbol> SymbolEnumerator;

   /** Enumerate all the globals known by this module.
      \note The enumerated symbol might be a GlobalSymbol or an UnknownSymbol.
    */
   void enumerateGlobals( SymbolEnumerator& rator ) const;

   /** Enumerate all exported global values known by this module.
          \note The enumerated symbol might be a GlobalSymbol or an UnknownSymbol.
    */
   void enumerateExports( SymbolEnumerator& rator ) const;

   /** Candy grammar to add exported functions. */
   Module& operator <<( Function* f )
   {
      addFunction( f );
      return *this;
   }

   /** Candy grammar to add exported classes. */
   Module& operator <<( Class* f )
   {
      addClass( f, false );
      return *this;
   }

   /** Adds a generic import request.
    \param source The source path or logical module name.
    \param bIsUri If true, the source is an URI, otherwise is a module name.
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

   uint32 lastGCMark() const { return m_lastGCMark; }

   
   /** Adds a standard import/from or load request.
    \param def An import definition.
    \return true if the import definition is compatible with the module,
    false if the symbols are already defined or if a load request was
    already issued for a newly loaded module.
    
    In case the function returns false, the caller should look into the
    import def and eventually raise a consistent error, then discard it.
    If it returns true, the ownership passes on the module.
    */
   Error* addImport( ImportDef* def );
   
   void removeImport( ImportDef* def );
   
   /** Shortcut to create a load ImportDef only if load is valid.
    \param name The name or URI of the module.
    \param bIsUri if name is URI.
    \return 0 if the load was already declared, a valid ImportDef 
    (already added to the module) if not.
    
    */
   ImportDef* addLoad( const String& name, bool bIsUri );
   
   /** Returns an imported symbol */
   Symbol* findImportedSymbol( const String& name ) const;

   /** Adds an implicitly imported symbol.
    \param uks The unknown symbol that should be resolved during link time.
    \param isNew True if the symbol is new.
    \return A valid symbol (possibly created as "extern" on the spot).
    
    Adds an implicit import. This is the same as import, but will always return
    a valid symbol, which might be undefined if the symbol wasn't known before,
    or it might be an existing global symbol.
    */
   Symbol* addImplicitImport( const String& name, bool& isNew );
   
   Symbol* addImplicitImport( const String& name )
   {
      bool bDummy;
      return addImplicitImport( name, bDummy );
   }
   
   /** Callback that is called when a symbol import request is satisfied.
    \param requester The module from which the request was issued.
    \param definer The module where the symbol has been found.
    \param sym The resolved symbol.
    */
   
   typedef Error* (*t_func_import_req)( Module* requester, Module* definer, Symbol* sym );
   
   /** Adds an import request.
    \param cbFunc a t_func_import_req callback that will be notified when this
      symbol is found (if ever). 
    \param symName the symbol to be searched.
    \param sourceMod The module that should provide the required symbol, or ""
           if the symbol should be searched in the global export table.
    \param bModIsPath True if sourceMod is a location or URI, false if it's
           a logical name.
               
    This method requests that a certain callback is invoked when a certain dependency is rsolved.
    */
   void addImportRequest( t_func_import_req cbFunc, const String& symName, 
               const String& sourceMod="", bool bModIsPath=false );
   
   /** Export a symbol.
    \param name The name of the symbol to be exported.
    \param bAlready will be set to true if the symbol was already defined.
    \return The exported symbol or 0 if the symbol was not defined.
    
    If the symbol is still not defined, zero is returned.
    
    \note it is NOT legal to export undefined symbols -- to avoid mistyping.
    */
   Symbol* addExport( const String& name, bool &bAlready );

   /** Adds an inheritance that's supposed to come from outside.
    \param inh The inheritance to be resolved during the link step.
    
    This method adds an inheritance of a FalconClass to the list of things
    to be resolved during the link phase.
    
    When all the external inheritances are resolved, the owner class is
    generated and readied to be used.
    
    \note this method creates an implicit import with the same name of the
            inherited class.
    \note This method must be called ONLY if the imported inheritance has not
            been found in the globals.
    */
   void addImportInheritance( Inheritance* inh );
   
   /** Adds an inheritance that is pending while the module is being formed.
    \param inh the inheritance to be added.
    
    In non-interactive compilation mode, it is posible to have forward declarations of 
    statically declared entities. When this happens with an inheritance clause,
    this fact must be specially remembered so that further declarations do not
    break the rules that the given symbol must be considered a class.
    
    Other methods checkPendingInheritance() and commitPendingiInheritances() are
    used to complete the controls on the inheritance system.
    */
   void addPendingInheritance( Inheritance* inh );
   
   /** Checks the symbol for existing inheritances.
      \param symName The name if the inheritance that has been created.
      \param parent The parent class, if defined.
      \return True if there are pending inheritances with this name, false otherwise.
    
    This method is called when statically defining functions or classes in non-interactive
    compilation. 
    
    In case the defined symbol is a class, the parent parameter should
    be the value of the newly defined classes; inheritances are then considered
    resolved, and they won't be added as external requests when the module 
    compilation is closed. 
    
    If they are function or objects, or any non-class statically declared
    entity, the parent parameter should be zero. In that case, the method will
    just check for the inheritance name to exist, and then return true. The
    compiler should add an error because a non-class entity has been used as
    static inheritance for a class.
    */
   bool checkPendingInheritance( const String& symName, Class* parent = 0 );
   
   /** Commits the inheritances into new external requests.
    
    To be called when the compilation of a module is complete. It will repeatedly
    call addImportInheritance on any existing inheritance.
   */
   void commitPendingInheritance();
   
   /** Adds a request for a foreign class not bound with an inheritance.
    \param cr A RquiredClass that will be filled with the required class.
    \return The symbol attached to this requirement (usuallyan undefined symbol).
    
    Classes inheriting from other classes are not the only statements
    specifically searching for classes in other modules. The RquiredClass class
    represents this fact, allowing statements, or generic third party modules,
    to ask for a foreign class.
    
    The caller should not generate a requirement for a global symbol
    that is already defined. However, if there is a global and defined symbol 
    matching the requirement, it is IMMEDIATELY resolved (Requirement::onResolved
    is called), and the globally defined symbol is returned.
    
    \note The method may throw immeately, as the requirement may throw immediately.
    
    */
   Symbol* addRequirement( Requirement* cr );
   
   
   /** Stores a class coming from a source module.
    \param fcls The class to be added.
    \param isObject if true, adds an object instance.
    \param gs An optional gobal symbol associated with this class.
    
    This method decides if a FalconClass should be added as complete (in this
    case, it may be transformed in a HyperClass if necessary) or if it requires
    to be posted for later linkage imports.
    
    Module compilers and code synthezizing classes that may require external
    inheritances should use this method.
    */
   void storeSourceClass( FalconClass* fcls, bool isObject, Symbol* gs = 0 );
   
   /** Perform completion checks on source classes.
    \param fcls The class that have just been completed.
    
    This method is called when all the inheritances of a FalconClass are
    (succesfully) resolved. In case it's necessary to create a HyperClass
    out of this FalconClass, the class ID and eventually the global symbol
    bound to this class are updated.
    */
   void completeClass( FalconClass* fcls );
   
   
   /** Reads the exportAll flag.
    \return True if this module wants to export all the non-private symbols.
    */
   bool exportAll() const { return m_bExportAll; }
   
   /** Sets the exportAll flag.
    \param e set to true if this module wants to export all the non-private symbols.
    */   
   void exportAll( bool e ) { m_bExportAll = e; }
   
   /** Unloads a dynamically loaded module.
    This method destroys the current module. In case the module has been
    crated through a dynamic shared object loader, the module is also unloaded.
    
    The system keeps a reference of loaded shared objects, so if the underlying
    shared library has been used to generate more modules, the other modules
    will still be able to work.
    
    */
   virtual void unload();
   

   /** Returns the module group associated with this module.    
    \return The module group in which this module is stored, or 0 if this
            is a static module living in the global module space.
    */
   ModSpace* moduleSpace() const { return m_modSpace; }
   
   void moduleSpace( ModSpace* md ) { m_modSpace = md; }
   
   bool isMain() const { return m_bMain; }
   
   void setMain( bool isMain = true ) { m_bMain = isMain; } 
   
   /** Executes a namespace forwarding from a module into this one. */
   void forwardNS( Module* mod, const String& remoteNS, const String& localNS );

   /** Check if we can use an existing module to resolve this requirement.
    
    We can use a module that has been already loaded if:
    - It's local (modgrouop), it has been imported privately and need to import it privately here.
    - It's local, and is imported publically or loaded (if not loaded, might get promoted).
    - It's global (modspace), and it's \b not imported privately here.
    
    Otherwise, a new load is needed (even if the module already exists somewhere.
    */
   Module* linkExistingModule( const String& name, bool bIsUri, t_loadMode imode );

   /** Searches a symbol, complete with its namespace in the import structure.
    \param name A symbol name, complete with its namespace prefix.
    \return A symbol if found, 0 otherwise.
    
    This method searches a symbol in the import structure of a module.
    
    The Symbol is not searched in the
    */
   Symbol* searchInImports( const String& name, Module*& mod );
   
   /** Returns a new default value stored in this module.
    \return A default value ready to be referenced by a symbol.
    
    Global, static and default parameter values are stored in the module
    and referenced by the symbols in their Symbol::defaultValue() field.
    
    This method creates a new default value that is stored in the module and
    stays valid as long as the module is alive. 
    
    \note Complex values as functions
    and classes properly reference back the module they come from, so the module
    stays alive as long as THEY are valid.
    */
   Item* addDefaultValue();
   
   /** Returns a new default value stored in this module.
    \param src The source value where to copy the item from.
    \return A default value ready to be referenced by a symbol.
    
    Global, static and default parameter values are stored in the module
    and referenced by the symbols in their Symbol::defaultValue() field.
    
    This method creates a new default value that is stored in the module and
    stays valid as long as the module is alive. 
    
    \note Complex values as functions
    and classes properly reference back the module they come from, so the module
    stays alive as long as THEY are valid.
    */
   Item* addDefaultValue( const Item& src );
   
   /** Adds a constant at global level in the module.
    \param name The name of the constant.
    \param value The value of the constant.
    \reutrn true if the name is free, false if the constant cannot be created.
    
    This method adds a constant for export in the module by adding a 
    global symbol and then assigning a default value to it.
    
    Assigmnet to the symbol is inhibited, but if the constant is complex (i.e.
    a class instance), it could be possible that its contents are actually
    changed.
    */
   bool addConstant( const String& name, const Item& value );
   
private:
   class Private;
   Private* _p;
   
   ModSpace* m_modSpace;
   String m_name;
   String m_uri;
   uint32 m_lastGCMark;
   bool m_bExportAll;
   DynUnloader* m_unloader;
   bool m_bMain;
   
   int m_anonFuncs;
   int m_anonClasses;
   
   friend class Private;   
   friend class DynLoader;
   friend class FAMLoader;
   friend class ModSpace;
   
   void name( const String& v ) { m_name = v; }
   void uri( const String& v ) { m_uri = v; }
   void setDynUnloader( DynUnloader* ul ) { m_unloader = ul; } 
   
   // checks for forward declarations, eventually removing them.
   void checkWaitingFwdDef( Symbol* sym );
   
   // used by various import and load requests.
   Error* addModuleRequirement( ImportDef* def );
   bool removeModuleRequirement( ImportDef* def );
};

}

#endif	/* MODULE_H */

/* end of module.h */

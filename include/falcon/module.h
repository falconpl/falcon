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
#include <falcon/enumerator.h>
#include <falcon/refcounter.h>
#include <falcon/modmap.h>
#include <falcon/loadmode.h>
#include <falcon/mantra.h>
#include <falcon/class.h>
#include <falcon/function.h>

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

   virtual ~Module();

   /** Logical name of the module. */
   const String& name() const { return m_name; }

   /** Physical world location of the module. */
   const String& uri() const {return m_uri;}


   /** Adds a global mantra, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A Symbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   Symbol* addMantra( Mantra* f, bool bExport = true );
   
   /** Storing it on an already defined symbol.
    \param sym The global symbol that is already stored on this module.
    \param f The function to be added

    
    */
   bool addMantraWithSymbol( Mantra* f, Symbol* sym, bool bExport = true );

   /** Adds an anonymous mantra.
    \param f The mantra to be added

    The name of the mantra will be modified so that it is unique in case
    it is already present in the module.
    
    The mantra will not be exported, and there won't be any synbol created
    for this mantra.
    */
   void addAnonMantra( Mantra* f );


   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.

    */
   Symbol* addFunction( const String& name, ext_func_t f, bool bExport = true );


   /** Creates a singleton object.
    \param fc The class to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   Symbol* addSingleton( Class* fc, bool bExport = true );


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
   Mantra* getMantra( const String& name, Mantra::t_category cat = Mantra::e_c_none ) const;

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
   Module& operator <<( Mantra* f )
   {
      addMantra( f );
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
   
   typedef Error* (*t_func_import_req)( Module* requester, const Module* definer, const Symbol* sym );
   
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
   
   void addImportRequest( Requirement* req, 
               const String& sourceMod="", bool bModIsPath=false );
   
   /** Export a symbol.
    \param name The name of the symbol to be exported.
    \param bAlready will be set to true if the symbol was already defined.
    \return The exported symbol or 0 if the symbol was not defined.
    
    If the symbol is still not defined, zero is returned.
    
    \note it is NOT legal to export undefined symbols -- to avoid mistyping.
    */
   Symbol* addExport( const String& name, bool &bAlready );   
   
   
   /** Adds a request for a foreign entity that shall be resolved at link phase.
    
    \param cr A Requirement that will be called back.
    \return The symbol attached to this requirement (usually an undefined symbol).       
    
    The caller should not generate a requirement for a global symbol
    that is already defined. However, if there is a global and defined symbol 
    matching the requirement, it is IMMEDIATELY resolved (Requirement::onResolved
    is called), and the globally defined symbol is returned.
    
    \note The method may throw immeately, as the requirement may throw immediately.
    
    \see Requirement
    */
   Symbol* addRequirement( Requirement* cr );

   
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
    
    The module space might be created either by the module itself,
    and used for private load, or be assigned by the loading process
    by other modules or by the loading module space.
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
   
   Function* getMainFunction();
   void setMainFunction( Function* mf );
   
   bool isNative() const { return m_bNative; }
   
   Class* getClass( const String& name ) const { return 
      static_cast<Class*>( getMantra(name, Mantra::e_c_class) ); }
      
   Function* getFunction( const String& name ) const { return 
      static_cast<Function*>( getMantra(name, Mantra::e_c_function) ); }
   
private:
   class Private;
   Module::Private* _p;
   
   ModSpace* m_modSpace;
   String m_name;
   String m_uri;
   uint32 m_lastGCMark;
   bool m_bExportAll;
   DynUnloader* m_unloader;
   bool m_bMain;
      
   int m_anonMantras;
   Function* m_mainFunc;
   bool m_bNative;

   friend class Private;   
   friend class DynLoader;
   friend class FAMLoader;
   friend class ModSpace;
   friend class ClassModule;
   
   void name( const String& v ) { m_name = v; }
   void uri( const String& v ) { m_uri = v; }
   void setDynUnloader( DynUnloader* ul ) { m_unloader = ul; } 
   
   // checks for forward declarations, eventually removing them.
   void checkWaitingFwdDef( Symbol* sym );
   
   // used by various import and load requests.
   Error* addModuleRequirement( ImportDef* def, ModRequest*& req );
   bool removeModuleRequirement( ImportDef* def );
   
   
   class FuncRequirement;      
};

}

#endif	/* MODULE_H */

/* end of module.h */

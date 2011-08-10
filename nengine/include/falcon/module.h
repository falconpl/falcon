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
#define	FALCON_MODULE_H

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

class GlobalSymbol;
class UnknownSymbol;
class Item;
class Class;
class Inheritance;
class UnknownSymbol;
class ModSpace;
class ModLoader;
class FalconClass;
class DynUnloader;
class ModGroup;

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
   void addStaticData( Class* cls, void* data );

   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   GlobalSymbol* addFunction( Function* f, bool bExport = true );

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
   void addFunction( GlobalSymbol* sym, Function* f );

   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.

    */
   GlobalSymbol* addFunction( const String& name, ext_func_t f, bool bExport = true );

   /** Adds a new class to this module.
    
    */
   void addClass( GlobalSymbol* gsym, Class* fc, bool isObj );

   /** Adds a global class, possibly exportable.
    \param fc The class to be added
    \param isObj If true, there's a singleton instance bound to this class.
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   GlobalSymbol* addClass( Class* fc, bool isObj, bool bExport = true );

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
   GlobalSymbol* addVariable( const String& name, bool bExport = true );

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
   GlobalSymbol* addVariable( const String& name, const Item& value, bool bExport = true );

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
   
   /** Enumerator receiving symbols in this module. */
   typedef Enumerator<UnknownSymbol> USymbolEnumerator;

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

   /** Sends dynamic data to the garbage. 
    This method is invoked as a part of the link step. When a DYNAMIC module
    is delivered to a virtual machine, it must send to the garbage collector
    all the static data that it does not want to account for. For instance,
    strings, ranges, and any deep data that may need collection but that is
    not going to keep this module alive when held somewhere.
    
    This method loops on all the static data and crates a garbage token
    for all those items that we don't want to keep.
    
    As classes and functions are bound to back-reference this module, and as
    they are destroyed as this module is dereferenced and disposed of, we won't
    create GC tokens for those entitites.
    */
   void sendDynamicToGarbage();

   /** Adds a load request.
    \param name The module to be loaded
    \param bIsUri Set to true if the module name is a system path or URI.
    */
   bool addLoad( const String& name, bool bIsUri=false );

   /** Adds a standard import/from request.
    \param localName The name of the symbol as it's locally known.
    \param remoteName The symbol name as it's known in the import source module.
    \parma source The import source module name or location.
    \param bIsUri Set to true if the module name is a system path or URI.
    */
   UnknownSymbol* addImportFrom( const String& localName, const String& remoteName,
                                           const String& source, bool bIsUri );

   /** Explicitly generate an imported global symbol.
    \param name The symbol to be searched.
    \return 0 if already existing, or a valid UnknownSymbol if not found.
    
    Generic imports are declared with "import from modulename [in namespace]".
    This method requries that all the unknown symbol (eventually related
    to a local namespace) are searched in the given module before being
    searched in the global export table.
        
    */
   UnknownSymbol* addImport( const String& name );
   
   /** Adds an implicitly imported symbol.
    \param uks The unknown symbol that should be resolved during link time.
    \return true if the symbol could be added, false if the symbol name was
    already existing in the module.
    */
   bool addImplicitImport( UnknownSymbol* uks );
   
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
    */
   void addImportInheritance( Inheritance* inh );
   
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
   void storeSourceClass( FalconClass* fcls, bool isObject, GlobalSymbol* gs = 0 );
   
   /** Perform completion checks on source classes.
    \param fcls The class that have just been completed.
    
    This method is called when all the inheritances of a FalconClass are
    (succesfully) resolved. In case it's necessary to create a HyperClass
    out of this FalconClass, the class ID and eventually the global symbol
    bound to this class are updated.
    */
   void completeClass( FalconClass* fcls );
   

   /** Resolves module requirements dynamically.
    \param loader A loader used to resolve dynamic deps.
    
    This puts all the resolved dependencies at disposal of this module.
    Resolved modules are destroyed when this module is destroyed.
    */
   bool resolveDynReqs( ModLoader* loader );
   
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
   
   /** Creates a namespace-sensible import.
    \TODO explain
    */
   bool addImportFromWithNS( const String& localNS, const String& remoteName, 
            const String& modName, bool isFsPath );

   /** Returns the module group associated with this module.    
    \return The module group in which this module is stored, or 0 if this
            is a static module living in the global module space.
    */
   ModGroup* moduleGroup() const { return m_modGroup; }
   
   void moduleGroup( ModGroup* md ) { m_modGroup = md; }
   
   bool isMain() const { return m_bMain; }
   
   void setMain( bool isMain = true ) { m_bMain = isMain; } 
   
   Error* resolveDirectImports( bool bUseModGroup );

   Error* exportToModspace( ModSpace* ms );
   
private:
   class Private;
   Private* _p;
   
   String m_name;
   String m_uri;
   uint32 m_lastGCMark;
   bool m_bExportAll;
   DynUnloader* m_unloader;
   ModGroup* m_modGroup;
   bool m_bMain;
   
   int m_anonFuncs;
   int m_anonClasses;
   
   friend class Private;   
   friend class DynLoader;
   friend class FAMLoader;
   friend class ModGroup;
   
   void name( const String& v ) { m_name = v; }
   void uri( const String& v ) { m_uri = v; }
   void setDynUnloader( DynUnloader* ul ) { m_unloader = ul; }
   
   Symbol* findGenericallyExportedSymbol( const String& symname, Module*& mod ) const;
   
   bool exportOnModGroup();
   
   void exportSymsInGroup( bool bForReal );
   void resolveGenericImports( bool bForReal );
   bool exportSymbolInGroup( Symbol* sym, bool bForReal );
   
   /** Resolve module requirements statically through a module space.
    
    This method tries to load or import all the modules from which this module
    depends.
    
      \note This is used directly and exclusively by the ModGroup class.
    
    As a module space is provided as a static module storage, the modules
    are loaded statically and added to the given modspace. As a result, they
    might automatically generate requests for other modules.
    
    Before trying to load a module, the ModSpace is searched. If a module is
    already available in the module space, it is used (and eventually promoted
    to "load requirement" if necessary), otherwise the ModLoader offered by
    the ModSpace is used to search for the module on the virtual file system.

    */
   bool resolveStaticReqs();
   
   /** Check if we can use an existing module to resolve this requirement.
    
    We can use a module that has been already loaded if:
    - It's local (modgrouop), it has been imported privately and need to import it privately here.
    - It's local, and is imported publically or loaded (if not loaded, might get promoted).
    - It's global (modspace), and it's \b not imported privately here.
    
    Otherwise, a new load is needed (even if the module already exists somewhere.
    */
   Module* linkExistingModule( const String& name, bool bIsUri, t_loadMode imode );

   Error* exportSymbolToModspace( ModSpace* ms, Symbol* sym );

};

}

#endif	/* MODULE_H */

/* end of module.h */

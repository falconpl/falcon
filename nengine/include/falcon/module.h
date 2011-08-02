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


   /** Mark (dynamic) modules for Garbage Collecting.
    \param mark the current GC mark.

    This is used by ClassModule when the module is handed to the Virtual
    Machine for dynamic collection.
    */
   void gcMark( uint32 mark ) { m_lastGCMark = mark; }

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

   bool addLoad( const String& name, bool bIsUri=false );

   UnknownSymbol* addImportFrom( const String& localName, const String& remoteName,
                                           const String& source, bool bIsUri );

   /** Explicitly generate an imported global symbol.
    \return 0 if already existing, or a valid UnknownSymbol if not found.
    */
   UnknownSymbol* addImport( const String& name );
   bool addImplicitImport( UnknownSymbol* uks );
   
   /** Export a symbol.
    \param name The name of the symbol to be exported.
    \param bAlready will be set to true if the symbol was already defined.
    \return The exported symbol or 0 if the symbol was not defined.
    
    If the symbol is still not defined, zero is returned.
    
    \note it is NOT legal to export undefined symbols -- to avoid mistyping.
    */
   Symbol* addExport( const String& name, bool &bAlready );

   void addImportInheritance( Inheritance* inh );
   
   /** Performs a passive link step on this module.
    During this step, the module tries to resolve all its own dependencies
    throught the modules and public symbols offered by the module space.
    
    This method is either called during the link phase of the module space
    (ModSpace::link()) or before injecting a dynamic module in the virtual 
    machine.
    */
   bool passiveLink( ModSpace* ms );
   
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
   
   /** Resolve module requirements statically through a module space.
    \param space The module space where dependencies are resolved statically.
    
    This method tries to load or import all the modules from which this module
    depends. 
    
    As a module space is provided as a static module storage, the modules
    are loaded statically and added to the given modspace. As a result, they
    might automatically generate requests for other modules.
    
    Before trying to load a module, the ModSpace is searched. If a module is
    already available in the module space, it is used (and eventually promoted
    to "load requirement" if necessary), otherwise the ModLoader offered by
    the ModSpace is used to search for the module on the virtual file system.

    \TODO more docs for throw
    */
   bool resolveStaticReqs( ModSpace* space );

   bool resolveDynReqs( ModLoader* loader );
   
   bool exportAll() const { return m_bExportAll; }
   void exportAll( bool e ) { m_bExportAll = e; }
   
private:
   String m_name;
   String m_uri;
   uint32 m_lastGCMark;
   bool m_bExportAll;

   class Private;
   Private* _p;
   friend class Private;   
};

}

#endif	/* MODULE_H */

/* end of module.h */

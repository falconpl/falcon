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
#include <falcon/module.h>
#include <falcon/vardatamap.h>
#include <falcon/attribute.h>

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
   Variable* addMantra( Mantra* f, bool bExport = true );
   
   /** Adds a class and the required structures to make it initializable at startup.
    *
    * Creates also an object entity to be initialized at module startup.
    *
    */
   Variable* addInitClass( Class* cls, bool bExport = true );

   /**
    * Returns the count of classes with an init-time instance to be filled.
    */
   int32 getInitCount() const ;

   /**
    * Returns the nth class with an init-time instance to be filled.
    */
   Class* getInitClass( int32 val ) const;


   Variable* addConstant( const String& name, const Item& value, bool bExport = true );

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
   Variable* addFunction( const String& name, ext_func_t f, bool bExport = true );


   /** Creates a singleton object.
    \param fc The class to be added
    \param bExport if true, the returned symbol will be exported.
    \return A variable containing the global ID or 0 if the variable
             was already declared.
    */
   Variable* addSingleton( Class* fc, bool bExport = true );


   /** Promotes a variable previously known as extern into a global.
    * \param ext The variable to be promoted.
    * \return true if the variable was an extern, false otherwise.
    *
    * This turns an extern variable into a global, eventually removing the
    * extern dependencies bound with the variable name.
    *
    * \note; on exit, the \b ext variable is extern.
    */
   bool promoteExtern( Variable* ext, const Item& value, int32 declaredAt );


   /** Sets an extern value once the relative extern variable is resolved.
    * \param name The name of the extern variable that is being resolved.
    * \param value The resolved value
    * \param source The source module where this value comes from (can be 0).
    * \return true if the variable name is found (and extern), false otherwise.
    *
    * If a dependency is waiting for this variable to be resolved, it is satisfied.
    * This might lead to throw an error if the satisfied depenency decides so.
    * */
   bool resolveExternValue( const String& name, Module* source, Item* value );

   /** Tries to create a new variable taking an external value.
    *
    * \param name the name for the extern global variable.
    * \param source A source module where this item is allocated (can be 0).
    * \param value The value to be associated with the variable.
    * \return A valid variable entry on success, 0 if the variable is already defined as global.
    *
    * If the variable is not defined, or defined as extern, the value is set.
    * This might also lead to dependency resolve, which in turn might raise an
    * error if the resolved dependency is willing to do so.
    *
    * If the value is already defined as a global variable, this method returns 0.
    */
   Variable* importValue( const String& name, Module* source, Item* value );


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
   Error* addLoad( const String& name, bool bIsUri );

   /** Adds an implicitly imported symbol.
    \param name The unknown symbol that should be resolved during link time.
    \param isNew True if the symbol is new.
    \return A valid variable (possibly created as "extern" on the spot).
    
    Adds an implicit import. This is the same as import, but will always return
    a valid variable, which might be undefined if the symbol wasn't known before,
    or it might be an existing global variable.
    */
   Variable* addImplicitImport( const String& name, bool& isNew );
   
   Variable* addImplicitImport( const String& name )
   {
      bool bDummy;
      return addImplicitImport( name, bDummy );
   }
   
   /** Removes an extern.
    * Mainly used by the interactive compiler to undo an unnecessary implicit import.
    *
    */
   bool removeExtern( const String& name );

   /** Callback that is called when a symbol import request is satisfied.
    \param requester The module from which the request was issued.
    \param definer The module where the symbol has been found.
    \param sym The resolved symbol.
    */
   
   typedef Error* (*t_func_import_req)( const Module* sourceModule, const String& sourceName, Module* targetModule, const Item& value, const Variable* targetVar );
   
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
   Variable* addRequirement( Requirement* cr );

   
   /** Perform completion checks on source classes.
    \param fcls The class that have just been completed.
    
    This method is called when all the inheritances of a FalconClass are
    (successfully) resolved. In case it's necessary to create a HyperClass
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
   uint32 depsCount() const;

   /** Get the nth import definition (dependency).
    * \param n Definition number in range 0 <= n < depsCount().
    * \return ImportDef for that number.
    */
   ImportDef* getDep( uint32 n ) const;

   VarDataMap& globals() { return m_globals; }
   const VarDataMap& globals() const { return m_globals; }


   Variable* addGlobal( const String& name, const Item& value, bool bExport = false )
   {
      VarDataMap::VarData* vd = m_globals.addGlobal(name, value, bExport);
      if( vd == 0 ) return 0;
      return &vd->m_var;
   }

   Variable* addExtern( const String& name, bool bExport = false )
   {
      VarDataMap::VarData* vd = m_globals.addExtern(name, bExport);
      if( vd == 0 ) return 0;
      return &vd->m_var;
   }


   Variable* addExport( const String& name, bool &bAlready )
   {
      VarDataMap::VarData* vd = m_globals.addExport(name, bAlready);
      if( vd == 0 ) return 0;
      return &vd->m_var;
   }

   inline Variable* addExport( const String& name ) {
       bool dummy;
       return addExport(name, dummy);
   }


   Item* getGlobalValue( const String& name ) const
   {
      return m_globals.getGlobalValue(name);
   }

   Item* getGlobalValue( uint32 id ) const
   {
      return m_globals.getGlobalValue(id);
   }

   Variable* getGlobal( const String& name ) const
   {
      VarDataMap::VarData* vd = m_globals.getGlobal(name);
      if( vd == 0 ) return 0;
      return &vd->m_var;
   }

   void exportNS( const String& sourceNS, Module* target, const String& targetNS )
   {
      m_globals.exportNS(this, sourceNS, target, targetNS );
   }

   const AttributeMap& attributes() const { return m_attributes; }
   AttributeMap& attributes() { return m_attributes; }

protected:
   /** Invoked when refcount hits 0.
    *  This will invoke the unload() method if not previously invoked.
    */
   virtual ~Module();

   /** Unloads a dynamically loaded module.
    This method destroys the current module. In case the module has been
    created through a dynamic shared object loader, the module is also unloaded.

    The system keeps a reference of loaded shared objects, so if the underlying
    shared library has been used to generate more modules, the other modules
    will still be able to work.

    */
   virtual void unload();

private:
   class Private;
   Module::Private* _p;
   
   VarDataMap m_globals;

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
   bool m_bLoad;

   AttributeMap m_attributes;

   friend class Private;   
   friend class DynLoader;
   friend class FAMLoader;
   friend class ModSpace;
   friend class ClassModule;
   
   void name( const String& v ) { m_name = v; }
   void uri( const String& v ) { m_uri = v; }
   void setDynUnloader( DynUnloader* ul ) { m_unloader = ul; } 
   
   // checks for forward declarations, eventually removing them.
   bool checkWaitingFwdDef( const String& name, Item* value );
   
   // used by various import and load requests.
   Error* addModuleRequirement( ImportDef* def, ModRequest*& req );
   bool removeModuleRequirement( ImportDef* def );

   class FuncRequirement;      

   FALCON_REFERENCECOUNT_DECLARE_INCDEC( Module );
};

}

#endif	/* MODULE_H */

/* end of module.h */

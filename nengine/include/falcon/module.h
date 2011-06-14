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

#include "enumerator.h"

namespace Falcon {

class GlobalSymbol;
class Item;

/** Standard Falcon Execution unit and library.

 Falcon modules are used to transiently store code and data that can then
 be executed by a virtual machine (usually, a single virtual machine, and just once).

 The contents of modules is dinamyc in the sense that it might be altered at
 any stage of their lifetime.

 Modules are divided mainly in two categories: static modules and dynamic modules
 (not to be confused with the fact that the module @b contents is dynamic).

 Static modules have a lifetime that is meant to be longer than that of the
 virtual machine they are attached to. They are injected into the virtual machine
 externally and cannot be explicitly referenced or unloaded by the owning VM.

 Dynamic modules can be loaded and unloaded multiple times during the lifetime
 of their own VM.

 The main differece between the two type of modules is that dynamic modules need
 to create entities that might surivive their own lifetime, and that need to be
 accounted separately for garbage reclaim.

 The advantage of declaring a module static is that all the items it declares
 are outside the scope of garbage collection. This has two effects: first,
 a program declaring a a static module and then feeding it in VM doesn't
 any special care about preventing that module to be destroyed by the garbage
 collector, and stays in control of its existence span. Second, the garbage
 collector is not required to perform useless checks on the items declared by
 the module (mainly functions and classes), that will be considered always
 valid.

 At VM level, items declared by static modules are considered UserItem instances,
 while items declared by dynamic modules are considered DeepItem instances.

 */
class FALCON_DYN_CLASS Module {
public:
   /** Creates an internal module.
    \param name The symbolic name of this module.
    \param bStatic True if this module is created as static.
    */
   Module( const String& name, bool bStatic = true );

   /** Creates an external module.
    \param name The symbolic name of this module.
    \param uri The uri from where this module was created.
    \param bStatic True if this module is created as static.
    */
   Module( const String& name, const String& uri, bool bStatic = true );

   virtual ~Module();

   const String& name() const { return m_name; }
   const String& uri() const {return m_uri;}


   /** Adds a global function, possibly exportable.
    \param f The function to be added
    \param bExport if true, the returned symbol will be exported.
    \return A GlobalSymbol holding a pointer to the global variable where the
            function is now stored, or 0 if the function name is already present.
    */
   GlobalSymbol* addFunction( Function* f, bool bExport = true );

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
    \return A global symbol or 0 if not found.
    
    If the given name is present as a global symbol in the current module.
    */
   GlobalSymbol* getGlobal( const String& name ) const;

   /** Finds a function.
    \param name The function name to be searched.
    \return A global function or 0 if not found.

    If the given name is present as a global function in the current module.
    */
   Function* getFunction( const String& name ) const;

   /** Enumerator receiving symbols in this module. */
   typedef Enumerator<GlobalSymbol*> SymbolEnumerator;

   /** Enumerate all the globals known by this module. */
   void enumerateGlobals( SymbolEnumerator& rator ) const;

   /** Enumerate all exported global values known by this module. */
   void enumerateExports( SymbolEnumerator& rator ) const;

   /** Candy grammar to add exported functions. */
   Module& operator <<( Function* f )
   {
      addFunction( f );
      return *this;
   }

public:
   String m_name;
   String m_uri;
   bool m_bIsStatic;

   class Private;
   Private* _p;
};

}

#endif	/* MODULE_H */

/* end of module.h */

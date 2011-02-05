/*
   FALCON - The Falcon Programming Language.
   FILE: function.h

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FUNCTION_H_
#define FALCON_FUNCTION_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/syntree.h>
#include <falcon/globalsvector.h>
#include <falcon/refpointer.h>


#include <map>
#include <vector>

namespace Falcon
{

class Item;
class Symbol;
class GlobalSymbol;
class Collector;

/**
 Falcon function.

 This class represents the minimal execution unit in Falcon. It's a set of
 code (to be excuted), symbols (parameters, local variables and reference to
 global variables in the module) and possibly closed values.

 Functions can be directly executed by the virtual machine.

 They usually reside in a module, of which they are able to access the global
 variable vector (and of which they keep a reference).

 To achieve higher performance, functions are not treated as
 normal garbageable items (the vast majority of them is never really
 destroyed). They become garbageable when their module is explicitly
 unloaded while linked, or when they are created dynamically as closures,
 or when constructed directly by the code.

 Functions can be created by modules or directly from the code. In this case,
 they aren't owned by any module and are immediately stored for garbage collection.

*/

class FALCON_DYN_CLASS Function
{
public:
   Function( const String& name, Module* owner = 0 );
   virtual ~Function();
   
   /** Sets the module of this function.
    Mainly, this information is used for debugging (i.e. to know where a function
    is declared).
    */
   void module( Module* owner );

   /** Return the module where this function is allocated.
   */
   Module* module() const { return m_module; }

   const String& name() const { return m_name; }

   Symbol* addVariable( const String& name );

   /** Adds a variable to with a value being provided from the outside.
    *
    * If the incoming value needs a deep copy, it should be performed
    * before passing it to this function.
    */
   Symbol* addClosedSymbol( const String& name, const Item& value );

   /** Gets the count of parameters in this function.
    * @return a number between 0..varCount()
    */

   int32 paramCount() const { return m_paramCount; }

   /** Sets the variable count.
    * @param pc The number of parameters in this functions.
    *
    * the parameter count should be a number in 0..varCount().
    */
   void paramCount( int32 pc ) { m_paramCount = pc; }

   /** Number of local variables in this function.
    * @return count of symbols declared in this function (including parameters).
    */
   int32 varCount() const { return m_locals.size(); }

   /** Finds a symbol by name. */
   Symbol* findSymbol( const String& name ) const;

   /** Gets a symbol by ID. */
   Symbol* getSymbol( int32 id ) const;

   /** Returns the statements of this function */
   const SynTree& syntree() const { return m_syntree; }
   SynTree& syntree() { return m_syntree; }

   /** Mark this function for garbage collecting. */
   void gcMark( int32 mark );

   /** Store in a garbage collector. 
    
    When this method is called, the function become subject to garbage
    collection.
   
    */
   void garbage( Collector* c );

   /** Gets the global vector associated with this function, if any.
      
       Only functions having a module can access a global vector.
    */
   GlobalsVector* globals();

protected:
   SynTree m_syntree;
   String m_name;
   int32 m_paramCount;

   GCToken* m_gcToken;   
   Module* m_module;

   //TODO: Use our old property table?
   // Anyhow, should be optimized a bit.
   typedef std::map<String, Symbol*> SymbolTable;
   SymbolTable m_symtabTable;

   typedef std::vector<Symbol*> SymbolVector;
   SymbolVector m_locals;

   ref_ptr<GlobalsVector> m_globals;
};

}

#endif /* FUNCTION_H_ */

/* end of function.h */

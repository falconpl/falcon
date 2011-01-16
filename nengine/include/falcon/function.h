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

#include <map>
#include <vector>

namespace Falcon
{

class Item;
class Symbol;

class FALCON_DYN_CLASS Function
{
public:
   Function( const String& name );
   virtual ~Function();

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

protected:
   SynTree m_syntree;
   String m_name;
   int32 m_paramCount;

   //TODO: Use our old property table?
   // Anyhow, should be optimized a bit.
   typedef std::map<String, Symbol*> SymbolTable;
   SymbolTable m_symtabTable;

   typedef std::vector<Symbol*> SymbolVector;
   SymbolVector m_locals;
};

}

#endif /* FUNCTION_H_ */

/* end of function.h */

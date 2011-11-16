/*
   FALCON - The Falcon Programming Language.
   FILE: symboltable.h

   Symbol table -- where to store local or global symbols.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 15:56:50 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_SYMBOLTABLE_H_
#define _FALCON_SYMBOLTABLE_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon
{

class String;
class Symbol;

/** Holder for symbols relative to the owning level.
 Symbols tables are found:
 - In functions, holding local symbols.
 - In modules, holding global symbols.
 - In the virtual machine, holding namespace-global symbols.

 Symbol tables are capable to hold local and non-local (global, dynamic or
 closed symbols).

 The difference between local symbols and other kind of symbols
 is that local symbols are always referencing a live data stack by displacement
 id; this means that the value of the variable they reference may change with
 the stack they refer to. Other kind of symbols have other variable mapping
 strategies that makes them context (or stack) independent.

 This requires local variables to be differently accounted, so that their users
 (mainly SynFunctions) can properly setup the stack before starting, and address
 them during their execution.

 As local symbols need separate accounting, the SymbolTable class has different
 methods to manage some of their specificity.

 Instead of creating two different classes to account for local and different
 symbols, the SymbolTable class can take care of both the kind of symbols. Some
 users (as the module or the VM) will use just some of the SymbolTable abilities,
 while Function SymbolTables won't (usually) own GlobalSymbol (but they can
 easily own ClosedSymbol).
 
 Dynamic symbols may be found in any symbol table.

 \note Once a symbol is assigned to a symbol table, it's owned by that table
 and destroyed with it.
*/

class FALCON_DYN_CLASS SymbolTable
{
public:
   SymbolTable();
   virtual ~SymbolTable();

   /** Number of local variables in this function.
      @return count of local symbols.

    \note This method is NOT an inline. Use sparcely.
    */
   int32 localCount() const;

   /** Finds a symbol by name.
    \param name The name of the symbol to be found.
    \return The symbol if found, or 0.

    The returned symbol may be either a local symbol or another kind of symbol
    known in the symbol table.
    */
   Symbol* findSymbol( const String& name ) const;

   /** Gets a local symbol by ID.
    \param id The id of the symbol (0 : localCount()-1).
    \return A symbol pointer or 0 if the id is out of range.

    */
  Symbol* getLocal( int32 id ) const;


   /** Adds a local symbol.
    \param The name of the local symbol.
    \return a Symbol newly created out of the given name, or an already
    existing symbol (warning: might not be a local symbol).

    If the name is already found int he local symbol table, that symbol is
    returned instead. So, the caller must take care to ascertain that the returned
    symbol is actually a local symbol before using it in a context where this
    difference matters.
    
    */
   Symbol* addLocal( const String& name );

   /** Adds an already created local symbol.
    \param sym The local symbol to be added to this table.
    \return True on success, false if the name is already used.
    
    If a symbol with the same name exists, the symbol is not added and the
    method returns false. The caller must then take proper action (i.e. destroy
    the symbol and signal error).

    \note The table takes ownership of this local symbol.
    */
   bool addLocal( Symbol* sym );

   /** Adds a non-local symbol to this table.
    \parma sym A symbol that is not considered local.
    \return True on success, false if the name is already used.

    If a symbol with the same name exists, the symbol is not added and the
    method returns false. The caller must then take proper action (i.e. destroy
    the symbol and signal error).

    \note The method doesn't check for the type of the incoming symbol, so it
    won't take any particular action if the incoming symbol is actually a
    LocalSymbol. The caller should separately call addLocal to update the local
    symbol count.
    */
   bool addSymbol( Symbol* sym );

private:
   class Private;
   Private* _p;
};

}

#endif /* _FALCON_SYMBOLTABLE_H_ */

/* end of synmboltable.h */

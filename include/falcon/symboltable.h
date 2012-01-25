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
class SynTree;
class Function;
class DataWriter;
class DataReader;

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
   SymbolTable( Function* parent );
   SymbolTable( SynTree* parent );
   SymbolTable( const SymbolTable& other);
   SymbolTable( SynTree* parent, const SymbolTable& other);
   SymbolTable( Function* parent, const SymbolTable& other);
   virtual ~SymbolTable();

   /** Number of local variables in this table.
      @return count of local symbols.

    \note This method is NOT an inline. Use sparcely.
    */
   int32 localCount() const;
   
   /** Number of closed variables in this table.
      @return count of closed symbols.

    \note This method is NOT an inline. Use sparcely.
    */
   int32 closedCount() const;

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
  
   /** Gets a closed symbol by ID.
    \param id The id of the symbol
    \return A symbol pointer or 0 if the id is out of range.

    */
  Symbol* getClosed( int32 id ) const;


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



   /** Adds a closed symbol.
    \param The name of the local symbol.
    \return a Symbol newly created out of the given name, or an already
    existing symbol (warning: might not be a local symbol).

    If the name is already found int he local symbol table, that symbol is
    returned instead. So, the caller must take care to ascertain that the returned
    symbol is actually a closed symbol before using it in a context where this
    difference matters.
    
    */
   Symbol* addClosed( const String& name );

   /** Sets a function as owner of this table.
    This is used only to retrieve a paret to mark if any of the owned symbol is marked.
    */
   void owner( SynTree* owner ) { m_owner.syntree = owner; m_ownedby = e_owned_syntree;}
   
   /** Sets a function as owner of this table.
    This is used only to retrieve a paret to mark if any of the owned symbol is marked.
    */
   void owner( Function* owner ) { m_owner.function = owner; m_ownedby = e_owned_function;}   
   
   /** Invoked by owned symbols when marked.
    */

   void gcMark( uint32 mark );
   
   void store( DataWriter* dw );
   void restore( DataReader* dr );
   
private:
   class Private;
   Private* _p;
   
   typedef enum {
      e_owned_none,
         e_owned_syntree,
         e_owned_function
   }
   t_ownedby;
   
   typedef union {
      SynTree* syntree;
      Function* function;
   }
   t_owner;
   
   t_owner m_owner;
   t_ownedby m_ownedby;
   
};

}

#endif /* _FALCON_SYMBOLTABLE_H_ */

/* end of synmboltable.h */

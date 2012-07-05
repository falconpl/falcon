/*
   FALCON - The Falcon Programming Language.
   FILE: sybmol.h

   Syntactic tree item definitions -- expression elements -- symbol.

   Pure virtual class base for the various symbol types.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_SYMBOL_H
#define FALCON_SYMBOL_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/item.h>
#include <falcon/variable.h>
#include <falcon/fassert.h>

#include "vmcontext.h"

namespace Falcon {

class VMContext;
class ExprSymbol;

/** Base symbol class.
 * A Falcon Symbol is a name indicating a value, possibly (usually) bound
 * with a Falcon item residing somewhere in the module, stack, virtual machine
 * or closed data table (for closures).
 *
 * Falcon symbols are divided into 4 categories.
 *
 * - Global symbols are linked either to the module or virtual machine
 *   data pool, and refer to a global namespace.
 * - Local symbols refer to items allocated in the stack at the nth position
 *   (either parameters or local variables).
 * - Closed symbols are linked with data explicitly closed in the sequence
 *   being currently executed (or in a parent sequence). Functions and code
 *   blocks are example of sequences.
 * - Dynamic symbols are names being resolved in the current execution context
 *   scanning the symbol table of the calling functions.
 *
 * As symbols are 0-ary expressions, they can be evaluated. Symbol evaluation
 * resolves into the item they are linked to, or in case of dynamic symbols
 * that cannot be bound, may cause an error to be raised.
 *
 * Symbol l-evaluation causes the linked item to be changed.
 *
 * As evaluation rules change depending on the symbol type, this is actually
 * an abstract class.
 *
 */
class FALCON_DYN_CLASS Symbol
{
public:
   typedef enum {
      e_st_local,
      e_st_global,
      e_st_closed,
      e_st_extern,
      e_st_dynamic
   } type_t;
   
   /** Value of the ID of undefined symbols. */
   const static uint32 undef=(uint32)-1;
   
   /** The default constructor creates a minimally configured undefined symbol
   */
   Symbol();

   /** Creates a dynamic symbol. */
   Symbol( const String& name, int line = 0 );
   
   /** Creates a generic symbol. */
   Symbol( const String& name, type_t type, uint32 asId, int line = 0 );
   
   /** Copies the other symbol */
   Symbol( const Symbol& other );
   
   ~Symbol();

   const String& name() const { return m_name; }
   void name( const String& n) { m_name = n; }
   type_t type() const { return m_type; }   

   /** Source line at which the symbol was declared.
    \return Source line in the file where this symbol was created, or 0 if not
    available or meaningless.
    */
   int32 declaredAt() const { return m_declaredAt; }
   /** Change the source line at which the symbol is created.
    \param l Source line in the file where this symbol was created, or 0 if not
    available or meaningless.
    */
   void declaredAt( int32 l ) { m_declaredAt = l; }
   
   /** Retreive the value associated with this symbol on this context.
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is undefined.
   */
   inline const Item* getValue( VMContext* ctx ) const { 
      return m_getVariable( const_cast<Symbol*>(this), ctx )->value(); }
   
   inline Item* lvalueValue( VMContext* ctx ) { return m_setVariable( this, ctx )->value(); }

   /** Retreive the value associated with this symbol on this context.
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is undefined.
   */
   inline const Variable* getVariable( VMContext* ctx ) const { return m_getVariable( const_cast<Symbol*>(this), ctx ); }
   /** Retreive the value associated with this symbol on this context (non-const).
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is undefined.
   */
   inline Variable* getVariable( VMContext* ctx ) { return m_getVariable( const_cast<Symbol*>(this), ctx ); }
   
   /** Retreive the value associated with this symbol, in L-value expressions.
    
    This method returns a variable associated with a symbol when the symbol
    is going to be assined a value (l-value expression). In case of dynsymbols,
    the resolution algorithm to find a value that is to be assigned is
    is fundamentally different (local binding) from the algorithm employed when
    the value is to be found for reference or read.
    
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is undefined.
   */
   inline Variable* lvalueVariable( VMContext* ctx ) { return m_setVariable( this, ctx ); }
   
   Symbol* clone() const { return new Symbol(*this); }
   
   void setConstant( bool bMode = true ) { m_bConstant = bMode; }
   bool isConstant() const { return m_bConstant; }
   
   void gcMark( uint32 mark );
   
   /** Gets the current mark of this symbol*/
   uint32 gcMark() const { return m_name.currentMark(); }
   
   /** Id of a local variable indicated by this symbol.
    -- Meaningful only if this symbol is local.
    */
   uint32 localId() const { return m_id; }
   void localId( uint32 id ) { m_id = id; }
   
   
   /** Invoked when an external reference is resolved onwards in a module.
    */
   void promoteToGlobal();
   
   /** Invoked by the linker when resolving an external value.
    */
   void resolved( Variable* other );

   /** Completely configure a global variable into this symbol.
    \param value The value given to the global symbol.
    */
   void globalWithValue( const Item& value );

   const Item& defaultValue() const {      
      return m_defValue;
   }
   
   Item& defaultValue() {      
      return m_defValue;
   }

   void defaultValue( const Item& item ) { 
      m_defValue = item;
   }

protected:
   
   // Notice: we're using the string GC mark to keep ours.
   String m_name;
   
   // source line at which they were declared
   int32 m_declaredAt;
   
   // Default value; used to initialize globals and params.   
   Item m_defValue;
   
   // Pointer to the real value of the symbol, used by globals and externs.
   Variable m_realValue;
   
   
   // ID used for local symbols.
   int32 m_id;
   
   Variable* (*m_getVariable)( Symbol* sym, VMContext* ctx );
   Variable* (*m_setVariable)( Symbol* sym, VMContext* ctx );
   
   type_t m_type;
   bool m_bConstant;
   
   static Variable* getVariable_global( Symbol* sym, VMContext* ctx );
   static Variable* getVariable_local( Symbol* sym, VMContext* ctx );
   static Variable* getVariable_closed( Symbol* sym, VMContext* ctx );
   static Variable* getVariable_extern( Symbol* sym, VMContext* ctx );
   static Variable* getVariable_dyns( Symbol* sym, VMContext* ctx );
   static Variable* setVariable_dyns( Symbol* sym, VMContext* ctx );

   friend class ExprSymbol;
};

}

#endif

/* end of symbol.h */

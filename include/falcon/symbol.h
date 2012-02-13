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
#include <falcon/fassert.h>

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
   inline const Item* getValue( VMContext* ctx ) const { return m_getValue( this, ctx ); }
   
   /** Sets the
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is read-only or undefineedd.
   */
   inline void setValue( VMContext* ctx, const Item& src ) const { m_setValue(this, ctx, src); }
   
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
   
   /** Promote an extern symbol. 
    \param other The imported symbol.
    
    An extern symbol evaluates to the value of the imported symbol.
    */
   void promoteExtern( Symbol* other )
   {
      fassert2( m_type == e_st_extern, "Ought to be an extern symbol." );
      fassert2(other->m_type == e_st_global, "Other symbol must be global." );
      
      m_other = other;
   }
   
   const Item& defaultValue() const {      
      return m_defValue;
   }

   Item& defaultValue() {      
      return m_defValue;
   }

   void defaultValue( const Item& item ) { 
      m_defValue = item;
   }
   
   /** Pointer to a symbol linekd as extern. */
   Symbol *externRef() const { return m_other; }
protected:
   
   // Notice: we're using the string GC mark to keep ours.
   String m_name;
   
   // source line at which they were declared
   int32 m_declaredAt;
   
   // Default value; can reference another symbol in extern symbols.   
   Item m_defValue;
   Symbol *m_other;
   
   // ID used for local symbols.
   int32 m_id;
   
   void (*m_setValue)( const Symbol* sym, VMContext* ctx, const Item& value );
   const Item* (*m_getValue)( const Symbol* sym, VMContext* ctx );
   
   type_t m_type;
   bool m_bConstant;
   
   static const Item* getValue_global( const Symbol* sym, VMContext* ctx );
   static const Item* getValue_local( const Symbol* sym, VMContext* ctx );
   static const Item* getValue_closed( const Symbol* sym, VMContext* ctx );
   static const Item* getValue_extern( const Symbol* sym, VMContext* ctx );
   static const Item* getValue_dyns( const Symbol* sym, VMContext* ctx );
   
   static void setValue_global( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_local( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_closed( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_extern( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_dyns( const Symbol* sym, VMContext* ctx, const Item& value );
   
   friend class ExprSymbol;
};

}

#endif

/* end of symbol.h */

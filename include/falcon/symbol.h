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

namespace Falcon {

class Stream;
class Item;
class Expression;
class VMContext;

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
 * TODO Serialization
 */
class FALCON_DYN_CLASS Symbol
{
public:
   typedef enum {
      e_st_local,
      e_st_global,
      e_st_closed,
      e_st_dynamic,
      e_st_extern,
      e_st_undefined
   } type_t;
   
   /** Value of the ID of undefined symbols. */
   const static uint32 undef=(uint32)-1;
   
   Symbol( const String& name, type_t t, uint32 id=undef );
   Symbol( const Symbol& other );
   ~Symbol();

   const String& name() const { return m_name; }
   type_t type() const { return m_type; }
   
   void define(type_t t, uint32 id);

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
    \return A pointer to the value exact location, or 0 if it cannot be
    determined.
   */
   Item* value( VMContext* ctx ) const { return m_value( ctx, this ); }
   
   /** Return the item.
    \return the ID of the symbol, or Symbol::undef if undefined.
    
    */
   uint32 id() const { return m_id; }
   
   /** Changes the id of this item.
    \param i The new ID for this symbol.
    */
   void id( uint32 i ) { m_id = i; }
   
   Symbol* clone() const { return new Symbol(*this); }
   
   /** Returns the default value of this symbol.
    \return the default value for this symbol or 0 if not set.
    
    Default values are assigned by setting a read-only value
    in a module or in a memory location outside the VM/GC scope.
    
    When they are linked, or in case of call, when they are pushed
    in the stack, this default value is applied.
    */
   Item* defaultValue() const { return m_defval; }
   
   /** Sets the default value for this symbol. 
    \param val The default value.
    
    The default value is not accounted, owned nor gc marked in any way.
    The caller must make sure that the item is properly gc'd or is outside
    from VM/GC action scope.
    */
   void defaultValue( Item* val ) { m_defval = val; }
   
   void setConstant( bool bMode = true ) { m_bConstant = bMode; }
   bool isConstant() const { return m_bConstant; }
   
protected:
   
   String m_name;
   // source line at which they were declared
   int32 m_declaredAt;         
   uint32 m_id;   
   type_t m_type;
   bool m_external;
   bool m_bConstant;
   
   Item* m_defval;

   Item* (*m_value)( VMContext* ctx, const Symbol* sym );
   static Item* value_global( VMContext* ctx, const Symbol* sym );
   static Item* value_local( VMContext* ctx, const Symbol* sym );
   static Item* value_closed( VMContext* ctx, const Symbol* sym );
   static Item* value_undef( VMContext* ctx, const Symbol* sym );
   static Item* value_dynamic( VMContext* ctx, const Symbol* sym );
};

}

#endif

/* end of symbol.h */

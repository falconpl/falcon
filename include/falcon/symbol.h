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

#include <falcon/fassert.h>
namespace Falcon {

class Item;
class Expression;
class VMContext;

class SymbolTable;
class Module;

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
   Symbol( const String& name, int line = 0);
   
   /** Creates a local symbol. */
   Symbol( const String& name, SymbolTable* host, uint32 asId, int line = 0 );
   
   /** Creates a global symbol. */
   Symbol( const String& name, Module* host, Item* value, int line = 0 );
   
   /** Creates an extern symbol. */
   static Symbol* ExternSymbol( const String& name, Module* mod, int line = 0 );
  
   /** Creates a closed symbol. */
   static Symbol* ClosedSymbol( const String& name, SymbolTable* host, uint32 id, int line = 0 );
   
   /** Copies the other symbol */
   Symbol( const Symbol& other );
   
   ~Symbol();

   const String& name() const { return m_name; }
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
   inline Item* getValue( VMContext* ctx ) const { return m_getValue( this, ctx ); }
   
   /** Sets the
    \param ctx The context where the symbol value is to be determined.
    \param target Where to store the symbol value.
    \throw An exception if the symbol is read-only or undefineedd.
   */
   inline void setValue( VMContext* ctx, const Item& src ) const { m_setValue(this, ctx, src); }
   
   Symbol* clone() const { return new Symbol(*this); }
   
   /** Returns the default value of this symbol.
    \return the default value for this symbol or 0 if not set.
    
    Default values are assigned by setting a read-only value
    in a module or in a memory location outside the VM/GC scope.
    
    When they are linked, or in case of call, when they are pushed
    in the stack, this default value is applied.
    */
   //Item* defaultValue() const { return m_defval; }
   
   /** Sets the default value for this symbol. 
    \param val The default value.
    
    The default value is not accounted, owned nor gc marked in any way.
    The caller must make sure that the item is properly gc'd or is outside
    from VM/GC action scope.
    */
   //void defaultValue( Item* val ) { m_defval = val; }
   
   void setConstant( bool bMode = true ) { m_bConstant = bMode; }
   bool isConstant() const { return m_bConstant; }
   
   void gcMark( uint32 mark );
   /** Gets the current mark of this symbol*/
   uint32 gcMark() const { return m_name.currentMark(); }
   
   /** Id of a local variable indicated by this symbol.
    -- Meaningful only if this symbol is local.
    */
   uint32 localId() const { return m_defvalue.asId; }
   
   /** Remote module in extern symbols. 
    This module is the module where the external global symbol is defined.
    */
   Module* remote() const { return m_remote; }
   
   /** Sets the remote module.
    This module is the module where the external global symbol is defined.
    */
   void remote( Module* rem ) { m_remote = rem; }
   
   /** Promote an extern symbol as "resolved" */
   void resolveExtern( Module* newHost, Item* value );
   
   /** Promote an extern symbol as global (forward declaration) */
   void promoteExtern( Item* value );
   
   Item* defaultValue() const { 
      fassert( m_type == e_st_global || m_type == e_st_extern );
      return m_defvalue.asItem;
   }
   
   void defaultValue( Item* item ) { 
      fassert( m_type == e_st_global || m_type == e_st_extern );
      m_defvalue.asItem = item;
   }
   
protected:
   typedef union {
      SymbolTable* symtab;
      Module* module;
   }
   t_host;
   
   // ID used for global and local symbols.
   typedef union {
      uint32 asId;
      Item* asItem;
   }
   t_defvalue;
   
   // Notice: we're using the string GC mark to keep ours.
   String m_name;
   // source line at which they were declared
   int32 m_declaredAt;         
   t_defvalue m_defvalue;      
   t_host m_host;
   
   // remote module filled when the other item is found.
   Module* m_remote;
   
   void (*m_setValue)( const Symbol* sym, VMContext* ctx, const Item& value );
   Item* (*m_getValue)( const Symbol* sym, VMContext* ctx );
   
   type_t m_type;
   bool m_bConstant;
   
   static Item* getValue_global( const Symbol* sym, VMContext* ctx );
   static Item* getValue_local( const Symbol* sym, VMContext* ctx );
   static Item* getValue_closed( const Symbol* sym, VMContext* ctx );
   static Item* getValue_extern( const Symbol* sym, VMContext* ctx );
   static Item* getValue_dyns( const Symbol* sym, VMContext* ctx );
   
   static void setValue_global( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_local( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_closed( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_extern( const Symbol* sym, VMContext* ctx, const Item& value );
   static void setValue_dyns( const Symbol* sym, VMContext* ctx, const Item& value );
};

}

#endif

/* end of symbol.h */

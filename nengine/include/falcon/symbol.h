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
      t_local_symbol,
      t_global_symbol,
      t_closed_symbol,
      t_dyn_symbol,
      t_unknown_symbol
   } type_t;

   Symbol( const Symbol& other );
   virtual ~Symbol();

   const String& name() const { return m_name; }

   /** Generates an expression for this symbol.
    * @return a new Expression to be set in a SynTree.
    */
   virtual Expression* makeExpression() = 0;

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

   /** Assign a value to this symbol.
    \param ctx the VM context where the symbol lives.
    \param value The value to be stored.
    This stores the value in the underlying item.
    Symbols not yet "living" in a virtual machine or in a readied module
    are not assignable; an exception would be raised in that case.
    */
   virtual void assign( VMContext* ctx, const Item& value ) const = 0;

   /** Assign a value to this symbol.
    \param value The value to be stored.
    \param ctx the VM context where the symbol lives.
    \param Return true if the symbol can be retreived; false if the symbol
    needs to live in context but the current context is not given.

    Gets the value associated with this symbol.
    
    \note This method is used during the link phase or when qyerying an
    exported symbol for the value that is being exported. The VM at run time
    uses a more sophisticated approach that obviates the need to call this
    virtual method. However, the result should be coherent.
    */
   virtual bool retrieve( Item& value, VMContext* ctx=0 ) const = 0;

protected:
   Symbol( type_t t, const String& name );

   type_t m_type;
   String m_name;
   // source line at which they were declared
   int32 m_declaredAt;
};

}

#endif

/* end of symbol.h */

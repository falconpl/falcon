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

protected:
   Symbol( type_t t, const String& name );

   type_t m_type;
   String m_name;
};

}

#endif

/* end of symbol.h */

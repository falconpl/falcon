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
#include <falcon/expression.h>

namespace Falcon {

class Stream;
class GarbageLock;
class VMachine;
class Item;

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
 */
class FALCON_DYN_CLASS Symbol: public Expression
{
public:
   Symbol( const Symbol& other );
   virtual ~Symbol();

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& result ) const { return false; }

   virtual void serialize( Stream* s ) const;
   virtual const String toString() const;
   const String& name() const { return m_name; }

   inline virtual void setLValue() { m_lvalue = true; }
   inline virtual bool isLValue() const { return m_lvalue; }

protected:
   Symbol( const String& name, t_operator type );
   virtual void deserialize( Stream* s );
   inline Symbol( t_operator type ):
      Expression( type )
   {}

   String m_name;
   bool m_lvalue;
   friend class ExprFactory;
};

}

#endif

/* end of symbol.h */

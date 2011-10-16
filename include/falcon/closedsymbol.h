/*
   FALCON - The Falcon Programming Language.
   FILE: closedsymbol.h

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLOSEDSYMBOL_H
#define FALCON_CLOSEDSYMBOL_H

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/item.h>

namespace Falcon {

class PStep;

/** Sybmols enclosing specific items in closures.
 *
 * These symbols hold a specific value which has been closed
 * (copied by value) from a enclosing context. GC Marking is
 * performed by the embedding item (a function or a sequence),
 * which has a separate map of closed symbols for this purpose.
 */
class FALCON_DYN_CLASS ClosedSymbol: public Symbol
{
public:
   ClosedSymbol( const String& name, const Item& closed );
   ClosedSymbol( const ClosedSymbol& other );
   virtual ~ClosedSymbol();

   static void apply_( const PStep*, VMContext* vm );
   static void apply_lvalue_( const PStep*, VMContext* vm );
   virtual ClosedSymbol* clone() const { return new ClosedSymbol(*this); }

   virtual Item* value( VMContext* ctx ) const;

   virtual Expression* makeExpression();

protected:
   Item m_item;
};

}

#endif

/* end of closedsymbol.h */

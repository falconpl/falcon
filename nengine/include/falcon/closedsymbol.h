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

namespace Falcon {

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
   ClosedSymbol( const String& name, const item& closed ):
      Symbol( name, t_closed_symbol ),
      m_item( closed )
   {}

   ClosedSymbol( const ClosedSymbol& other );
   virtual ~ClosedSymbol();

   virtual void evaluate( VMachine* vm, Item& value ) const;
   virtual void leval( VMachine* vm, const Item& assignand, Item& value );

   virtual void serialize( Stream* s ) const;

   ClosedSymbol* clone() const { return new ClosedSymbol(*this); }

   const Item& value() const { return m_item; }
   Item& value() { return m_item; }
protected:
   virtual void deserialize( Stream* s );
   inline ClosedSymbol():
      Symbol( t_closed_symbol )
   {}

   Item m_item;
   friend class ExprFactory;
};

}

#endif

/* end of closedsymbol.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: dynsymbol.h

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DYNSYMBOL_H
#define FALCON_DYNSYMBOL_H

#include <falcon/setup.h>
#include <falcon/symbol.h>

namespace Falcon {

/** Symbols that must be resolved at runtime.
 */
class FALCON_DYN_CLASS DynSymbol: public Symbol
{
public:
   DynSymbol( const String& name, const item& closed ):
      Symbol( name, t_dyn_symbol )
   {}

   DynSymbol( const DynSymbol& other );
   virtual ~DynSymbol();

   virtual void evaluate( VMachine* vm, Item& value ) const;
   virtual void leval( VMachine* vm, const Item& assignand, Item& value );

   virtual void serialize( Stream* s ) const;

   DynSymbol* clone() const { return new DynSymbol(*this); }

protected:
   virtual void deserialize( Stream* s );
   inline DynSymbol():
      DynSymbol( t_closed_symbol )
   {}

   friend class ExprFactory;
};

}

#endif

/* end of dynsymbol.h */

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
   DynSymbol( const String& name ):
      Symbol( t_dyn_symbol, name )
   {}

   DynSymbol( const DynSymbol& other );
   virtual ~DynSymbol();

   virtual void apply( VMachine* vm ) const;
   virtual void serialize( Stream* s ) const;

   DynSymbol* clone() const { return new DynSymbol(*this); }

protected:
   virtual void deserialize( Stream* s );
   inline DynSymbol():
      Symbol( t_dyn_symbol )
   {}

   friend class ExprFactory;
};

}

#endif

/* end of dynsymbol.h */

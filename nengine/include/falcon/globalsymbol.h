/*
   FALCON - The Falcon Programming Language.
   FILE: globalsybmol.h

   Syntactic tree item definitions -- expression elements -- symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_GLOBALSYMBOL_H
#define FALCON_GLOBALSYMBOL_H

#include <falcon/symbol.h>

namespace Falcon {

/** Global symbol class.
 *
 * This class references a value in the module symbol table.
 * Symbols may be imported from other modules as well. In that case,
 * they are marked as "external" and resolved by the VM at link time.
 *
 * The import process generates a common global reference for exported
 * or trans-module items, so that they get marked as long as there are
 * modules referencing them.
 */
class FALCON_DYN_CLASS GlobalSymbol: public Symbol
{
public:
   GlobalSymbol( const String& name, Item* itemPtr );
   GlobalSymbol( const GlobalSymbol& other );
   virtual ~GlobalSymbol();

   virtual GlobalSymbol* clone() const { return new GlobalSymbol(*this); }
   virtual void serialize( Stream* s ) const;

   virtual void apply( VMachine* vm ) const;

protected:
   virtual void deserialize( Stream* s );
   inline GlobalSymbol():
      Symbol( t_global_symbol )
   {}

   Item* m_itemPtr;
   friend class ExprFactory;
};

}

#endif

/* end of globalsymbol.h */

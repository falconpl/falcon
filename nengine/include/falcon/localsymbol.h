/*
   FALCON - The Falcon Programming Language.
   FILE: localsybmol.h

   Syntactic tree item definitions -- expression elements -- local symbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Jan 2011 18:46:25 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_LOCALSYMBOL_H
#define FALCON_LOCALSYMBOL_H

#include <falcon/setup.h>
#include <falcon/symbol.h>

namespace Falcon {

class FALCON_DYN_CLASS LocalSymbol: public Symbol
{
public:
   LocalSymbol( const String& name, int id ):
      Symbol( name, t_local_symbol ),
      m_id( id )
   {}

   LocalSymbol( const LocalSymbol& other );
   virtual ~LocalSymbol();

   virtual void perform( VMachine* vm ) const;
   virtual void apply( VMachine* vm ) const;

   virtual void serialize( Stream* s ) const;
   LocalSymbol* clone() const { return new LocalSymbol(*this); }

protected:
   virtual void deserialize( Stream* s );
   inline LocalSymbol():
      Symbol( t_local_symbol )
   {}

   int m_id;
   friend class ExprFactory;
};

}

#endif

/* end of localsymbol.h */

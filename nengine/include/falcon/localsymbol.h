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

class PStep;
class SymbolTable;
class VMContext;
class VMachine;

class FALCON_DYN_CLASS LocalSymbol: public Symbol
{
public:
   LocalSymbol(const String& name, int id );
   LocalSymbol( const LocalSymbol& other );
   virtual ~LocalSymbol();

   virtual LocalSymbol* clone() const { return new LocalSymbol(*this); }

   virtual void assign( VMContext* ctx, const Item& item ) const;
   virtual bool retrieve( Item& value, VMContext* ctx ) const;
   
   static void apply_( const PStep* s1, VMContext* ctx );

   virtual Expression* makeExpression();

   int id() const { return m_id; }

protected:
   friend class SymbolTable;
   int m_id;
};

}

#endif

/* end of localsymbol.h */

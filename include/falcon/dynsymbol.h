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

class PStep;
class VMContext;
class VMachine;

/** Symbols that must be resolved at runtime.
 */
class FALCON_DYN_CLASS DynSymbol: public Symbol
{
public:
   DynSymbol( const String& name );
   DynSymbol( const DynSymbol& other );
   virtual ~DynSymbol();

   virtual DynSymbol* clone() const { return new DynSymbol(*this); }

   virtual Item* value( VMContext* ctx ) const;
   static void apply_( const PStep* self, VMContext* ctx );
   static void apply_lvalue_( const PStep* self, VMContext* ctx );
   virtual Expression* makeExpression();

protected:
   DynSymbol();
};

}

#endif

/* end of dynsymbol.h */

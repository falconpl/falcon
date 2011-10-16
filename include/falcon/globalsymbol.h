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
#include <falcon/item.h>

namespace Falcon {

class PStep;

/** Global symbol class.

 This class references a value in the module symbol table.
 Symbols may be imported from other modules as well. In that case,
 they are marked as "external" and resolved by the VM at link time.
 
 */
class FALCON_DYN_CLASS GlobalSymbol: public Symbol
{
public:
   GlobalSymbol( const String& name );
   GlobalSymbol( const String& name, const Item& initValue );
   GlobalSymbol( const GlobalSymbol& other );   
   virtual ~GlobalSymbol();

   virtual GlobalSymbol* clone() const { return new GlobalSymbol(*this); }
   
   static void apply_( const PStep* self, VMContext* ctx );
   static void apply_lvalue_( const PStep* self, VMContext* ctx );
   virtual Expression* makeExpression();

   virtual Item* value( VMContext* ) const;

protected:
   GlobalSymbol();
   Item m_item;
};

}

#endif

/* end of globalsymbol.h */

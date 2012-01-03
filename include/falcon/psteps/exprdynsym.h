/*
   FALCON - The Falcon Programming Language.
   FILE: exprdynsym.h

   Syntactic tree item definitions -- expression elements -- dynsymbol.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 23:00:05 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRDYNSYM_H
#define FALCON_EXPRDYNSYM_H

#include <falcon/setup.h>
#include <falcon/expression.h>

namespace Falcon {

class DynSymbol;

/** Expression referring to a symbol.
 
 */
class FALCON_DYN_CLASS ExprDynSymbol: public Expression
{
public:
   ExprDynSymbol( int line = 0, int chr = 0 );
   
   /** Declare a forward symbol reference.
    \param name The name of the symbol that should be used in this expression.
    
    This constrcutor can be used to create a symbol placeholder in an expression,
    which can be then filled later on.
    */
   ExprDynSymbol( DynSymbol* sym, int line = 0, int chr = 0 );
   
   ExprDynSymbol( const ExprDynSymbol& other );
   virtual ~ExprDynSymbol();

   virtual void describeTo(String &str, int depth=0) const;

   inline virtual ExprDynSymbol* clone() const { return new ExprDynSymbol(*this); }

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& ) const { return false; }
   inline virtual bool isStatic() const { return false; }

   // Return the symbol pointed by this expression.
   DynSymbol* dynSymbol() const { return m_symbol; }
   void dynSymbol( DynSymbol* sym ) { m_symbol = sym; }


protected:
   DynSymbol* m_symbol;
   
   static void apply_( const PStep* ps, VMContext* ctx );
   
   class FALCON_DYN_CLASS PStepLValue: public PStep
   {
   public:
      ExprDynSymbol* m_owner;
      
      PStepLValue( ExprDynSymbol* owner ): m_owner(owner) { apply = apply_; }
      virtual void describeTo( String&, int depth=0 ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };   
   
   PStepLValue m_pslv;
};

}

#endif

/* end of exprdynsym.h */

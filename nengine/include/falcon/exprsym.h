/*
   FALCON - The Falcon Programming Language.
   FILE: exprsym.h

   Syntactic tree item definitions -- expression elements -- symbol.

   Pure virtual class base for the various symbol types.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 12:23:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRSYM_H
#define FALCON_EXPRSYM_H

#include <falcon/setup.h>
#include <falcon/expression.h>

namespace Falcon {

class Symbol;
class DynSymbol;
class GlobalSymbol;
class LocalSymbol;
class ClosedSymbol;


/** Expression referring to a symbol.
 *
 * These expressions have the role to access a symbol in read
 * or write (lvalue) mode in an expression..
 *
 *
 */
class FALCON_DYN_CLASS ExprSymbol: public Expression
{
public:
   ExprSymbol( const ExprSymbol& other );
   virtual ~ExprSymbol();

   virtual void describe(String & str) const;

   inline virtual ExprSymbol* clone() const { return new ExprSymbol(*this); }

   /** Symbols cannot be simplified. */
   inline virtual bool simplify( Item& ) const { return false; }
   inline virtual bool isStatic() const { return false; }

   virtual void serialize( DataWriter* s ) const;

   // Return the symbol pointed by this expression.
   Symbol* symbol() const { return m_symbol; }
   void symbol(Symbol* sym) { m_symbol = sym; }


   /** Redefine precompile in lvalue context.
    
    */
   void precompileLvalue( PCode* pcode ) const;

protected:
   
   class PStepLValue: public PStep
   {
   public:
      ExprSymbol* m_owner;
      
      PStepLValue( ExprSymbol* owner ): m_owner(owner) {}
      virtual void describe( String& ) const;
      
   };   
   PStepLValue m_pslv;
   
   ExprSymbol( Symbol* target );

   virtual void deserialize( DataReader* s );
   inline ExprSymbol( operator_t type ):
      Expression( type ),
      m_pslv(this)
   {}

   /** Used by the symbol classes to set the adequate handler function. */
   void setApply( apply_func func ) { apply = func; }
   
   /** Used by the symbol classes to set the adequate handler function. */
   void setApplyLvalue( apply_func func ) { m_pslv.apply = func; }

   Symbol* m_symbol;

   friend class ExprFactory;
   friend class DynSymbol;
   friend class GlobalSymbol;
   friend class LocalSymbol;
   friend class ClosedSymbol;
   friend class UnknownSymbol;
};

}

#endif

/* end of exprsym.h */

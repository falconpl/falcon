/*
   FALCON - The Falcon Programming Language.
   FILE: expriif.h

   Syntactic tree item definitions -- Ternary fast-if expression
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 Oct 2011 21:30:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRIIF_H
#define FALCON_EXPRIIF_H

#include <falcon/expression.h>

namespace Falcon {

/** Fast if -- ternary conditional operator. */
class FALCON_DYN_CLASS ExprIIF: public TernaryExpression
{
public:
   ExprIIF( int line =0, int chr = 0 );
   ExprIIF( Expression* op1, Expression* op2, Expression* op3, int line =0, int chr = 0 );
   ExprIIF( const ExprIIF& other );
   virtual ~ExprIIF();

   inline virtual ExprIIF* clone() const { return new ExprIIF( *this ); }
   virtual bool simplify( Item& value ) const;
   static void apply_( const PStep*, VMContext* ctx );
   virtual void render( TextWriter* tw, int depth ) const;

   /** Check if the and expression can stand alone.
      An "?" expression can stand alone if the second AND third operand are standalone.
    */
   inline virtual bool isStandAlone() const {
      return m_second->isStandAlone() && m_third->isStandAlone();
   }

private:

   class FALCON_DYN_CLASS Gate: public PStep {
   public:
      Gate( ExprIIF* owner );
      virtual void describeTo( String& target ) const { target = "Gate for expriif"; }
      static void apply_( const PStep*, VMContext* ctx );
   private:
      ExprIIF* m_owner;
   } m_gate;
};

}

#endif	/* EXPRIIF_H */

/* end of expriif.h */

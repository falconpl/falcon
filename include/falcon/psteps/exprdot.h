/*
   FALCON - The Falcon Programming Language.
   FILE: exprdot.h

   Syntactic tree item definitions -- Dot accessor
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Bgin: Sat, 30 Jul 2011 16:26:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRDOT_H_
#define _FALCON_EXPRDOT_H_

#include <falcon/expression.h>

namespace Falcon
{

/** Dot accessor. */
class FALCON_DYN_CLASS ExprDot: public UnaryExpression
{
public:
   ExprDot( const String& prop, Expression* op1, int line = 0, int chr = 0 );
   ExprDot( int line=0, int chr=0 );      
   ExprDot( const ExprDot& other );
   
   virtual ~ExprDot();
   
   inline virtual ExprDot* clone() const { return new ExprDot( *this ); } 
   virtual bool simplify( Item& value ) const; 
   static void apply_( const PStep*, VMContext* ctx );
   virtual void describeTo( String&, int depth=0 ) const;
   
   const String& property() const { return m_prop; }
   void property( const String& p ) { m_prop = p; }
   
protected:
   class FALCON_DYN_CLASS PstepLValue: public PStep
   {
   public:
      PstepLValue( ExprDot* owner ): m_owner(owner) { apply = apply_; }
      virtual ~PstepLValue() {}
      static void apply_( const PStep*, VMContext* ctx );
   private:
      ExprDot* m_owner;
   };
   PstepLValue m_pslv;
   
   String m_prop;
};

}

#endif

/* end of exprdot.h */

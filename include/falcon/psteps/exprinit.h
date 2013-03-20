/*
   FALCON - The Falcon Programming Language.
   FILE: exprinit.h

   Syntactic tree item definitions -- Init values for generators
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Feb 2013 18:11:20 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXPRINIT_H_
#define FALCON_EXPRINIT_H_

#include <falcon/expression.h>

namespace Falcon {

/** Class implementing init value for generators.
 */
class FALCON_DYN_CLASS ExprInit: public Expression
{
public:
   ExprInit( int line = 0, int chr = 0 );
   ExprInit( const ExprInit &other );
   virtual ~ExprInit();

   virtual bool isStatic() const;
   virtual ExprInit* clone() const { return new ExprInit(*this); }
   virtual bool simplify( Item& ) const { return false; }
   virtual void render( TextWriter* tw, int depth ) const;

private:
   static void apply_( const PStep* s1, VMContext* ctx );

   class FALCON_DYN_CLASS PStepLValue: public PStep
   {
   public:
      PStepLValue(){ apply = apply_; }
      virtual ~PStepLValue(){}
      virtual void describeTo( String& ) const;
      static void apply_( const PStep* ps, VMContext* ctx );
   };

   PStepLValue m_pslv;
};

}

#endif

/* end of exprinit.h */

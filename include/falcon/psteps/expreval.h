/*
   FALCON - The Falcon Programming Language.
   FILE: expreval.h

   Evaluation expression (^* expr) 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 20:02:46 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPREVAL_H_
#define _FALCON_EXPREVAL_H_

#include <falcon/setup.h>
#include <falcon/expression.h>

namespace Falcon {

/** Evaluation expression (^* expr) 
 */
class ExprEval: public UnaryExpression
{
public:
   ExprEval();
   ExprEval( Expression* expr );
   ExprEval( const ExprEval& other );
   
   virtual ~ExprEval() {};   
    
   virtual void describeTo( String&, int depth = 0 ) const;
    
   virtual Expression* clone() const { return new ExprEval(*this); }
   inline virtual bool isStandAlone() const { return true; }
   virtual bool isStatic() const {return false; }
   virtual bool simplify( Item& ) const { return false; }      
   
public:
   /** Placeholder for break after having pushed our body.*/
   class PStepResetOC: public PStep 
   {
   public:
      PStepResetOC() { apply = apply_; m_bIsFinally = true; }
      virtual ~PStepResetOC() {}
      virtual void describeTo( String& tgt, int =0 ) const { tgt = "Reset OC evaluation"; }
      
   private:
      static void apply_( const PStep*, VMContext* ctx );
   }
   m_resetOC;   
   
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif	/* _FALCON_EXPREVAL_H_ */

/* end of expreval.h */

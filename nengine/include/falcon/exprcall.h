/*
   FALCON - The Falcon Programming Language.
   FILE: exprcall.cpp

   Expression controlling item (function) call
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 21:19:26 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRCALL_H_
#define _FALCON_EXPRCALL_H_

#include <falcon/expression.h>

#include "pseudofunc.h"

namespace Falcon {

class FALCON_DYN_CLASS ExprCall: public Expression
{
public:
   ExprCall( Expression* op1 );

   /** Create a call-through-pseudo function.
    Calls through pseudofunctions are performed by pushing the
    pseudofunction PStep instead of using this expression psteps.
    */
   ExprCall( PseudoFunction* func );

   ExprCall( const ExprCall& other );
   virtual ~ExprCall();

   inline virtual ExprCall* clone() const { return new ExprCall( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void describe( String& ) const;
   virtual void oneLiner( String& s ) const { describe( s ); }
   inline String describe() const { return PStep::describe(); }
   inline String oneLiner() const { return PStep::oneLiner(); }

   int paramCount() const;
   Expression* getParam( int n ) const;
   ExprCall& addParam( Expression* );

   inline virtual bool isStandAlone() const { return false; }
   void precompile( PCode* pcode ) const;

   virtual bool isBinaryOperator() const { return false; }

   virtual bool isStatic() const { return false; }


   /** Returns the pseudofunction associated with this call.
    \return Pseudofunction associated with this expression, or 0 if none.

    If this expression call is actually calling a pseudofunction,
    this will return a non-zero pointer to a PseudoFunction stored
    in the Engine.
    */
   PseudoFunction* pseudo() const { return m_func; }

protected:
   inline ExprCall();
   friend class ExprFactory;
   PseudoFunction* m_func;
   Expression* m_callExpr;

private:
   class Private;
   Private* _p;

   // PStep used to push a pseudofunction when used in function mode.
   class FALCON_DYN_CLASS PStepPushFunc: public PStep
   {
   public:
      PStepPushFunc( PseudoFunction* func ):
         m_func( func )
      {
         apply = apply_;
      }

      virtual void describe( String& txt ) const;
      static void apply_( const PStep* ps, VMContext* ctx );

   private:
      PseudoFunction* m_func;
   };

   PStepPushFunc m_psPushFunc;

   static void apply_( const PStep*, VMContext* ctx );
   static void apply_dummy_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprcall.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: exprpseudocall.h

   Expression controlling pseudofunction call
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 13 Jan 2012 12:46:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_EXPRPSEUDOCALL_H_
#define _FALCON_EXPRPSEUDOCALL_H_

#include <falcon/setup.h>
#include <falcon/pseudofunc.h>
#include <falcon/psteps/exprvector.h>

namespace Falcon {

/** Expression calling a pseudofunction.
 
 This expressions is specifically created to call a certain pseudofunction.
 If the pseudofunction is called through its standard parameters, the invocation
 is performed directly as a VM PStep invocation, otherwise the function is
 called as a ususal by creating a call frame and generating a Function::invoke()
 call.
 
 If the pseudofunction is an ETA, its parameters are not evaluated.
 
 \note The function is considered to be safely stored and no mean is taken to 
 gc-mark it (pseudofunction are generally stored in the engine or genearted
 in embedding application, and have lifespan equal or greater than the engine).
 
 */
class FALCON_DYN_CLASS ExprPseudoCall: public ExprVector
{
public:
   ExprPseudoCall( int line=0, int chr=0 );

   /** Create a call-through-pseudo function.
    Calls through pseudofunctions are performed by pushing the
    pseudofunction PStep instead of using this expression psteps.
    */
   ExprPseudoCall( PseudoFunction* func, int line=0, int chr=0 );

   ExprPseudoCall( const ExprPseudoCall& other );
   virtual ~ExprPseudoCall();

   inline virtual ExprPseudoCall* clone() const { return new ExprPseudoCall( *this ); }
   virtual bool simplify( Item& value ) const;
   virtual void render( TextWriter* tw, int32 depth ) const;

   inline virtual bool isStandAlone() const { return false; }

   virtual bool isStatic() const { return false; }

   /** Returns the pseudofunction associated with this call.
    \return Pseudofunction associated with this expression.
    */
   PseudoFunction* pseudo() const { return m_func; }
   
   /** Associates a new pseudofunction with this call.
    \param ps The new pseudofunction.
    */
   void pseudo(PseudoFunction* ps );

protected:
   PseudoFunction* m_func;
   
private:  
   static void apply_( const PStep*, VMContext* ctx );
   static void apply_eta_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of exprpseudocall.h */

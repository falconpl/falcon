/*
   FALCON - The Falcon Programming Language.
   FILE: pseudofunc.h

   Pseudo function definition and standard pseudo functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 19:05:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef _FALCON_PSEUDOFUNC_H
#define	_FALCON_PSEUDOFUNC_H

#include <falcon/function.h>
#include <falcon/pstep.h>

namespace Falcon {

/** Pseudo function base class.

 Pseudo functions are functions that can be inserted by the compiler directly
 at their spot when they are found and are consistent with their declaration.
 In short, they provide a PStep that can be inserted directly as a part of an
 expression when they are found, so that they don't receive a call frame, nor
 expect one.

 PseudoFunction instances are derived from Function; this means that they
 can be considered functions under many aspects; for example, they can be
 referenced as function items and be normally called, for example, when
 being processed in a functional sequence or when assigned to a variable and
 then called at a later moment. However, they offer also the ability to be
 called directly by the vm from within the flow of an expression, without
 the need (and the overhead) of creating a call frame and returning from it.

 A typical example is "max()", which is often implemented in scripting langauges
 as a sort of compile-time macro or built-in function to determine the highest
 of two values. In case of Falcon, the following code:

 @code
 if max(a,b) > 100: printl( "High value" )
 @endcode

 will expand max into a pseudocode not involving any VM-level function call
 frame, while

 @code
 reduce( max, .[ 1 2 3 4 ] )
 @endcode

 will call iteratively the "max" pseudofunction as any other Falcon function.

 \note To be expanded at syntactic level when possible, Pseudo functions must
 be published in the Engine via Engine::publishPseudoFunction.

 \note subclasses must implement the pstep() method, returning the PStep
 instance that this class wants to publish to the compiler.
 */
class FALCON_DYN_CLASS PseudoFunction: public Function
{
public:
   PseudoFunction( const String& name, PStep* invoke_step );
   
   virtual ~PseudoFunction();
   
   /** Returns the PStep invoked that invokes this function from within an expression.
    \return The PStep instance that invokes this functions.

    The ownership of the returned pointer stays on this function; this means that
    this method should return an object in this class.
    */
   const PStep* pstep() const { return m_step; }

private:
   PStep* m_step;
};

}

#endif	/* _FALCON_PSEUDOFUNC_H */

/* end of pseudofunc.h */
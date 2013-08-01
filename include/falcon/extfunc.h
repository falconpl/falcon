/*
   FALCON - The Falcon Programming Language.
   FILE: extfunc.h

   Definition for the external function type
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_EXTFUNC_H_
#define FALCON_EXTFUNC_H_

#include <falcon/function.h>

namespace Falcon
{

/** Marker that can be used to declare a Falcon extension function. */
#ifndef FALCON_FUNC
#define FALCON_FUNC void
#endif

/**
   External function helper.

   This helper is a function just calling an immediate external function.

   The called function must not re-enter the virtual machine (i.e. call another
 Falcon script function via the VMachine::call method). The invoke() method
 of this class automatically calls performs the following tasks:
 
 - clears the A register on input.
 - invokes the function
 - calls the return frame
 - stores the value of the A register on top of the stack, substituting the
   item that was right before the parameters of the call.
 
 This means that the extension function is considered terminated as soon as it
 returns, and cannot re-enter the virtual machine or ask for further PSteps
 to be executed.

 In case calls to other VM Functions are needed, it's adviasble to re-extend
 Function for a finer control of the execution process.
*/

class FALCON_DYN_CLASS ExtFunc: public Function
{
public:
   ExtFunc( const String& name, ext_func_t func, Module* owner = 0, int32 line = 0 ):
      Function( name, owner, line ),
      m_func(func)
   {}

   ExtFunc( const String& name, const String& desc, ext_func_t func, Module* owner = 0, int32 line = 0 ):
      Function( name, owner, line ),
      m_func(func)
   {
      parseDescription(desc);
   }

   virtual ~ExtFunc() {}
   virtual void invoke( VMContext* ctx, int32 pCount = 0 );

protected:
   ext_func_t m_func;
};

}
#endif /* FALCON_EXTFUNC_H_ */

/* end of extfunc.h */

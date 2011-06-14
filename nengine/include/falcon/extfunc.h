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

/** External function type */
typedef void (*ext_func_t)(VMachine*);

/** Marker that can be used to declare a Falcon extension function. */
#ifndef FALCON_FUNC
#define FALCON_FUNC void
#endif

/**
   External function helper.

   This helper is a function just calling an immediate external function.

   The called function must not re-enter the virtual machine (i.e. call another
 Falcon script function via the VMachine::call method). The apply() method
 of this class automatically calls the VMachine::returnFrame() as soon as the
 extension function exits. This means that the extension function is considered
 terminated as soon as it returns.

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
   virtual ~ExtFunc() {}
   virtual void apply( VMachine* vm, int32 pCount = 0 );

protected:
   ext_func_t m_func;
};

}
#endif /* FALCON_EXTFUNC_H_ */

/* end of extfunc.h */

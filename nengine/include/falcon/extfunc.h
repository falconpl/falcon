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

/**
   External function helper.

   This helper is a function just calling an immediate external function.

   The called function must not re-enter the virtual machine, or if it does,
   it must push a proper static PStep and then return.

 A function pushing itself this way should also be pushed in the stack for
 proper garbage collecting prevention.
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

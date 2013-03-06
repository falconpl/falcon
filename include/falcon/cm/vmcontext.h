/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.h

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_VMCONTEXT_H
#define FALCON_CORE_VMCONTEXT_H

#include <falcon/fassert.h>
#include <falcon/cm/vmcontextbase.h>

namespace Falcon {
namespace Ext {

/*#
 @object VMContext
 @brief public interface to the VMContext class.

 An instance of this class gives access to the execution context
 where it is created.

 @code
    > VMContext.id         // current context id
    > VMContext.callDepth  // call count
    > VMContext.caller(0)  // current called object, equivalent to fself
    > VMContext.caller(1)  // caller object...
 @endcode

 @prop id Unique ID of the context.
 @prop callDepth Number of callers above the current level.
 @prop processId Unique ID of the process owning this context.
 @prop dataDepth Size of the data stack
 @prop codeDepth Size of the code stack.
 @prop selfItem Equivalent to self.
 */
class ClassVMContext: public ClassVMContextBase
{
public:
   
   ClassVMContext();
   virtual ~ClassVMContext();
   
   //=============================================================
   //
   virtual void* createInstance() const;
};

}
}

#endif

/* end of vmcontext.h */

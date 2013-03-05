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
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

/*#
 @class VMContext
 @brief Reflective inspector of the current execution context.

 An instance of this class gives access to the execution context
 where it is created.

 @code
    ctx = VMContext()
    > ctx.id()   // current context id
    > ctx.callDepth()  // call count
    > ctx.caller(0)    // current called object, equivalent to fself
    > ctx.caller(1)    // caller object...
 @endcode

 @prop id Unique ID of the context.
 @prop callDepth Number of callers above the current level.
 @prop processId Unique ID of the process owning this context.
 @prop dataDepth Size of the data stack
 @prop codeDepth Size of the code stack.
 @prop selfItem Equivalent to self.
 */
class ClassVMContext: public Class
{
public:
   
   ClassVMContext();
   virtual ~ClassVMContext();
   
   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;
   virtual void op_toString( VMContext* ctx, void* instance ) const;

};

}
}

#endif

/* end of vmcontext.h */

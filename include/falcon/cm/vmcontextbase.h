/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontextbase.h

   Falcon core module -- Interface to the vmcontext class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_VMCONTEXT_BASE_H
#define FALCON_CORE_VMCONTEXT_BASE_H

#include <falcon/fassert.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

/*#
 @class #VMContext
 @brief Reflective inspector of the current execution context.
 @see VMContext

   This class is accessible through a singleton only,
   named @a VMContext. See that singleton for further information.
 */
class ClassVMContextBase: public Class
{
public:
   
   ClassVMContextBase();
   virtual ~ClassVMContextBase();
   
   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;


   virtual void op_toString( VMContext* ctx, void* instance ) const;

protected:
   ClassVMContextBase( const String& name );

private:
   void init();
};

}
}

#endif

/* end of vmcontext.h */

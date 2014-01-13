/*
   FALCON - The Falcon Programming Language.
   FILE: vmcontext.h

   Falcon core module -- Interface to the collector.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 06 Mar 2013 17:40:44 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_VMPROCESS_H
#define FALCON_CORE_VMPROCESS_H

#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/collectoralgorithm.h>

namespace Falcon {
class PStep;

namespace Ext {


/*#
 @class VMProcess 
 @brief Access to sub-processes in the virtual machine.

 @prop stdIn Standard input bound with the process.
 @prop stdOut Standard output bound with the process.
 @prop stdErr Standard error stream bound with the process.
 @prop current Current process (static).
 */
class ClassVMProcess: public Class
{
public:
   
   ClassVMProcess();
   virtual ~ClassVMProcess();
   
   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

   PStep* m_stepAfterPersist;
};

}
}

#endif

/* end of gc.h */

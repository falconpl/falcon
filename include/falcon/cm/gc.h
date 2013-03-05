/*
   FALCON - The Falcon Programming Language.
   FILE: gc.h

   Falcon core module -- Interface to the collector.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 26 Jan 2013 19:35:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_GC_H
#define FALCON_CORE_GC_H

#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/collectoralgorithm.h>

namespace Falcon {
namespace Ext {

/*#
 @object GC
 @brief Controls the garbage collector.

 @prop memory Memory currently controlled and supposed alive by the collector
 @prop items Number of items currently controlled and supposed alive by the collector
 @prop enabled True if the collector works, false to disable the collector.
 @prop status current GC status: 0=green, 1=yellow, 2=red
 @prop marks Contexts marked since last time @a GC.reset() was invoked
 @prop sweeps Count of sweep loops since last time @a GC.reset() was invoked
 */
class ClassGC: public Class
{
public:
   
   ClassGC();
   virtual ~ClassGC();
   
   //=============================================================
   //
   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;
};

}
}

#endif

/* end of gc.h */

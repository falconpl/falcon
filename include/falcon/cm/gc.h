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
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/classes/classuser.h>

namespace Falcon {
namespace Ext {

/*#
 @object GC
 @brief Controls the garbage collector.

 @prop memory Memory currently controlled and supposed alive by the collector
 @prop items Number of items currently controlled and supposed alive by the collector
 @prop enabled True if the collector works, false to disable the collector.
 @prop current GC status: 0=green, 1=yellow, 2=red
 */
class ClassGC: public ClassUser
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
private:   
   
   //====================================================
   // Properties.
   //
   
   FALCON_DECLARE_PROPERTY( memory )
   FALCON_DECLARE_PROPERTY( items )
   FALCON_DECLARE_PROPERTY( enabled )
   FALCON_DECLARE_PROPERTY( status )

   /*#
    @method perform GC
    @brief Suggests or forces a full garbage collecting.
    @optparam force True to ask for a total garbage collection.
    @optparam wait True to wait until the collection is complete.

    If @b force is false, then the current context only is scheduled for
    inspection as soon as possible. This can cause a delay in the
    execution of subsequent instructions. However, the calling context
    might not see the memory immediately freed, as reclaim happens
    at a later stage.

    If @b force is true, then all the existing contexts are marked
    for inspection, and inspected as soon as possible. If @b wait
    is also true, then the calling context stays blocked until all
    the currently existing contexts are checked, and all the garbage
    memory is actually reclaimed.

    @note If @b force is false, @b wait is ignored. To see memory
    effectively reclaimed in a single agent application after this
    call, set both parameters to true nevertheless.
    */
   FALCON_DECLARE_METHOD( perform, "force:[B], wait:[B]" );

   /*#
    @method suggest GC
    @brief Invites the GC to inspect the oldest or all the contexts
    @optparam all True to ask for inspection on all the contexts.

    The method returns immediately; the GC will try to collect
    the available memory as soon as possible.
    */
   FALCON_DECLARE_METHOD( suggest, "all:[B]" );
};

}
}

#endif

/* end of gc.h */

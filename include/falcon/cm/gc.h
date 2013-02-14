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

   /*#
    @method caller VMContext
    @brief Returns the item (function or method) that is calling the current function.
    @optparam depth If specified, return the nth parameter up to @a VMContext.codeDepth

    If @b depth is not specified, it defaults to 1. Using 0 returns the same entity as
    obtained by the @b fself keyword.
    */
   //FALCON_DECLARE_METHOD( caller, "depth:[N]" );
};

}
}

#endif

/* end of gc.h */

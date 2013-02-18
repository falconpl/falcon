/*
   FALCON - The Falcon Programming Language.
   FILE: waiter.h

   Falcon core module -- Object helping to wait on repeated shared
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 29 Nov 2012 13:52:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_WAITER_H_
#define _FALCON_WAITER_H_

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/classes/classuser.h>
#include <falcon/property.h>
#include <falcon/method.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#

*/
class ClassWaiter: public ClassUser
{
public:
   ClassWaiter();
   virtual ~ClassWaiter();

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* insatnce ) const;
   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   void store( VMContext*, DataWriter*, void* ) const;
   void restore( VMContext*, DataReader*) const;
   void flatten( VMContext*, ItemArray&, void* ) const;
   void unflatten( VMContext*, ItemArray&, void* ) const;

   //=============================================================
   //
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;
   virtual void op_in( VMContext* ctx, void* instance ) const;

private:

   FALCON_DECLARE_PROPERTY( len );

   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
   FALCON_DECLARE_METHOD( tryWait, "" );
   FALCON_DECLARE_METHOD( add, "shared:Shared" );
   FALCON_DECLARE_METHOD( remove, "shared:Shared" );
};

}
}


#endif

/* end of waiter.h */

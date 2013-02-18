/*
   FALCON - The Falcon Programming Language.
   FILE: semaphore.h

   Falcon core module -- Semaphore shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_SEMAPHORE_H
#define FALCON_CORE_SEMAPHORE_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/method.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedSemaphore: public Shared
{
public:
   SharedSemaphore( const Class* owner, int32 initCount );
   virtual ~SharedSemaphore();
};


class FALCON_DYN_CLASS ClassSemaphore: public ClassShared
{
public:
   ClassSemaphore();
   virtual ~ClassSemaphore();

   //=============================================================
   //
   virtual void* createInstance() const;
   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;

private:

   FALCON_DECLARE_METHOD( post, "count:[N]" );
   FALCON_DECLARE_METHOD( wait, "timeout:[N]" );
};

}


}

#endif	

/* end of semaphore.h */

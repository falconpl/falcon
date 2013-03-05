/*
   FALCON - The Falcon Programming Language.
   FILE: mutex.h

   Falcon core module -- Barrier shared object
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2013 22:34:32 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_MUTEX_H
#define FALCON_CORE_MUTEX_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/string.h>
#include <falcon/shared.h>
#include <falcon/classes/classshared.h>

#include <falcon/atomic.h>

#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS SharedMutex: public Shared
{
public:
   SharedMutex( ContextManager* mgr, const Class* owner );
   virtual ~SharedMutex();

   virtual int32 consumeSignal( VMContext*, int32 count = 1 );

   void addLock();
   int32 removeLock();

protected:
   virtual int32 lockedConsumeSignal( VMContext*, int32 count );

private:
   atomic_int m_count;
};

/*#
  @class Mutex
  @brief A classical reentrant mutual-exclusion device.
  @ingroup parallel

 */
class FALCON_DYN_CLASS ClassMutex: public ClassShared
{
public:
   ClassMutex();
   virtual ~ClassMutex();

   //=============================================================
   //
   virtual void* createInstance() const;

   virtual bool op_init( VMContext* ctx, void*, int pcount ) const;


   FALCON_DECLARE_INTERNAL_PSTEP( UnlockAndReturn );
};

}
}

#endif

/* end of mutex.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: shared_ipsem.h

   Interface for the Falcon VM to a shared IP semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_SHAREDIPSEM_H_
#define _FALCON_FEATHERS_SHAREDIPSEM_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/shared.h>

#include "ipsem.h"

namespace Falcon {

/** Interface for the Virtual Machine to an interprocess semaphore.
 *
 * While the IPSem class abstracts the system-level IP semaphore, this
 * class wraps the abstraction into a structure that can be used as a
 * shared object by the Falcon VM (e.g. scripts).
 *
 */
class SharedIPSem: public Shared
{
public:
   SharedIPSem( ContextManager* ctx, Class* handler );
   SharedIPSem( ContextManager* ctx, Class* handler, const String& name );
   SharedIPSem( const IPSem& other );

   virtual ~SharedIPSem();

   IPSem& semaphore() { return m_sem; }
   const IPSem& semaphore() const { return m_sem; }

   virtual int32 consumeSignal( VMContext* target, int32 count = 1 );

private:
   class Private;
   Private* _p;

   IPSem m_sem;
};

}

/* end of shared_ipsem.h */
#endif /* _FALCON_FEATHERS_IPSEM_H_ */

/*
   FALCON - The Falcon Programming Language.
   FILE: ipsem_ext.h

   Inter-process semaphore.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 11 Nov 2013 16:27:30 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FEATHERS_IPSEM_EXT_H_
#define _FALCON_FEATHERS_IPSEM_EXT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

namespace {
class SemWaiter;
}

/** Interface for the IPSem
 *
 */
class ClassIPSem: public Class
{
public:
   ClassIPSem();
   virtual ~ClassIPSem();

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual int64 occupiedMemory( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   /** Used by an internal class to know if it should stay active or not. */
   bool checkNeeded( SemWaiter* threadData ) const;

private:
   class Private;
   Private* _p;
};

}

#endif /* _FALCON_FEATHERS_IPSEM_EXT_H_ */

/* end of ipsem_ext.h */

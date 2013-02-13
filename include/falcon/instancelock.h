/*
   FALCON - The Falcon Programming Language.
   FILE: instancelock.h

   Generic instance-wide lock
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 12 Feb 2013 23:51:42 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_INSTANCELOCK_H_
#define _FALCON_INSTANCELOCK_H_

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

/** Generic instance-wide lock
 *
 *
 * This class can be used by the Class handlers or third party users whenever
 * an instance has to be accessed exclusively, in case the instance doesn't
 * have any direct mean to be locked, and it is not convenient
 * to create a carrier hosting a mutex.
 *
*/

class FALCON_DYN_SYM InstanceLock
{
public:
   InstanceLock();
   ~InstanceLock();

   class Token;

   /** Generic instance-wide lock.
    *
    */
   Token* lock( void* instance ) const;

   /** Try Generic instance-wide lock.
    *
    */
   Token* trylock( void* instance ) const;

   /** Generic instance-wide unlock.
    *
    */
   void unlock( Token* li ) const;

   class Locker {
   public:
      Locker( const InstanceLock* lk, void* instance ):
         m_il(lk),
         m_token( lk->lock(instance) )
         {}

      ~Locker() {
         m_il->unlock(m_token);
      }

   private:
      const InstanceLock* m_il;
      InstanceLock::Token* m_token;
   };

private:
   class Private;
   Private* _p;
};

}

#endif

/* end of instancelock.h */

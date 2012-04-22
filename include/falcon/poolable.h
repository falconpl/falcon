/*
   FALCON - The Falcon Programming Language.
   FILE: poolable.h

   Generic pool for recycleable instances. 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 21 Apr 2012 21:07:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_POOLABLE_H
#define FALCON_POOLABLE_H

#include <falcon/pool.h>

namespace Falcon {


class Poolable
{
public:
   inline void dispose() { m_pool->release( this ); }
   inline void assignToPool( Pool* p ) { m_pool = p; }
   
protected:
   virtual ~Poolable() {};
   
private:
   Poolable* m_next;
   Pool* m_pool;
   
   friend class Pool;
};

}

#endif	/* FALCON_POOLABLE_H */

/* end of poolable.h */

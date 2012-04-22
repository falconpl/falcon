/*
   FALCON - The Falcon Programming Language.
   FILE: pool.h

   Generic pool for recycleable instances. 
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 21 Apr 2012 21:07:10 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_POOL_H
#define FALCON_POOL_H

#include <falcon/mt.h>

namespace Falcon {

class Poolable;

/** Generic pool for recycleable instances. 
 
 This class has basic support for recycling commonly used small data slices.
 
*/
class Pool 
{
public:
   Pool( uint32 maxSize = (uint32)-1 );
   inline virtual ~Pool() { clear(); }
   
   uint32 size();
   void clear();
   void release( Poolable* data );
   Poolable* get();
   
private:
   Poolable* m_head;
   uint32 m_size;
   uint32 m_maxSize;
   Mutex m_mtx;
};

}

#endif	/* POOL_H */

/* end of pool.h */

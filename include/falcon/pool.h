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

/** Generic pool for recyclable instances.
 
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
   
   template<class _T>
   _T* tget() { return static_cast<_T*>(get()); }

   template<class _T>
   _T* xget() {
      _T* value = static_cast<_T*>(get());
      if( value == 0 )
      {
         value = new _T;
         value->assignToPool(this);
      }
      return value;
   }

private:
   Poolable* m_head;
   uint32 m_size;
   uint32 m_maxSize;
   Mutex m_mtx;
};

/** Generic pool FIFO list structure.

 This class has basic support for enqueueing and recycling commonly used small data slices.

*/
class PoolFIFO
{
public:
   PoolFIFO();
   inline virtual ~PoolFIFO() { clear(); }

   void enqueue(Poolable* element);
   Poolable* dequeue();
   bool empty() const { return m_head == 0; }
   void clear();

   template<class _T>
   _T* tdeq() { return static_cast<_T*>(dequeue()); }

private:
   Poolable* m_head;
   Poolable* m_tail;
   mutable Mutex m_mtx;
};


}

#endif	/* POOL_H */

/* end of pool.h */

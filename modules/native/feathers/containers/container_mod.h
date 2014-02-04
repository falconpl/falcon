/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: container.h
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_FEATHERS_CONTAINERS_CONTAINER_H_
#define _FALCON_FEATHERS_CONTAINERS_CONTAINER_H_

#include <falcon/mt.h>
#include <falcon/atomic.h>
#include <falcon/item.h>

namespace Falcon {
class ClassContainerBase;

namespace Mod {
class Iterator;

/** Base class for all containers. */
class Container
{
public:
   Container( const ClassContainerBase* h );
   virtual ~Container();

   virtual bool empty() const = 0;
   virtual int64 size() const = 0;
   virtual Iterator* iterator() = 0;
   virtual Iterator* riterator() = 0;
   virtual Container* clone() const = 0;
   virtual void append(const Item& value ) = 0;
   virtual void clear() = 0;
   virtual bool erase(Iterator* iter) = 0;
   virtual bool insert(Iterator* iter, const Item& value ) = 0;

   /** Check if the container contains the item, OR prepares the context to do the checking.
    *
    * \return true if the container contains the value, false if it doesn't or if the context is prepared.
    *
    * On false return, the caller must check the depth of the context code; if changed, then
    * the VM must be run to check the container.
    */
   virtual bool contains( VMContext* ctx, const Item& value );
   virtual void gcMark( uint32 m );

   uint32 currentMark() const { return m_mark; }
   void lock() const { m_mtx.lock(); }
   void unlock() const { m_mtx.unlock(); }

   int32 version() const { return m_version; }

   const ClassContainerBase* handler() const { return m_handler; }

protected:
   int32 m_version;

private:
   uint32 m_mark;
   mutable Mutex m_mtx;
   const ClassContainerBase* m_handler;
};

}
}

#endif

/* end of container.h */

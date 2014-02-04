/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: iterator.h
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

#ifndef _FALCON_FEATHERS_CONTAINERS_ITERATOR_H_
#define _FALCON_FEATHERS_CONTAINERS_ITERATOR_H_

#include <falcon/class.h>

namespace Falcon {
namespace Mod {
class Container;

/** Iterator base class.
 * Operations on iterators throw a InvariantError in case the underling container changes.
 *
 */
class Iterator
{
public:
   Iterator( Container* c, bool rev ): m_bReverse(rev), m_container(c), m_mark(0) {}
   Iterator( const  Iterator& c ): m_bReverse(c.m_bReverse), m_container(c.m_container), m_mark(0) {}
   virtual ~Iterator();

   virtual bool next(Item& value) = 0;
   virtual bool hasNext() = 0;
   virtual bool current(Item& value) = 0;
   virtual void reset() = 0;
   virtual Iterator* clone() const = 0;

   Container* container() const { return m_container; }
   void gcMark( uint32 m );
   uint32 currentMark() const { return m_mark; }
   bool isReverse() const { return m_bReverse; }

protected:
   bool m_bReverse;

private:
   Container* m_container;
   uint32 m_mark;

};
}
}

#endif

/* end of iterator.h */


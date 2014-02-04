/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: list_mod.h
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

#ifndef _FALCON_FEATHERS_CONTAINERS_LIST_MOD_H_
#define _FALCON_FEATHERS_CONTAINERS_LIST_MOD_H_

#include "container_mod.h"

namespace Falcon {
namespace Mod {


class List: public Container
{
public:

   static const uint32 DEFAULT_POOL_SIZE = 256;
   List(const ClassContainerBase* cls, uint32 poolSize = DEFAULT_POOL_SIZE );
   List(const List& other);
   virtual ~List();

   virtual bool empty() const;
   virtual int64 size() const;
   virtual Iterator* iterator();
   virtual Iterator* riterator();
   virtual Container* clone() const;
   virtual void append(const Item& value );
   virtual void clear();
   virtual bool erase(Iterator* iter);
   virtual bool insert(Iterator* iter, const Item& value );

   void poolSize( uint32 size );
   uint32 poolSize();

   void push(const Item& item);
   void pop();
   void unshift(const Item& item);
   void shift();

   bool front( Item& target ) const;
   bool back( Item& target ) const;

   class Private;
   Private* prv() const { return _p; }

private:
   int64 m_size;
   Private* _p;
};
}

}

#endif

/* end of list_mod.h */


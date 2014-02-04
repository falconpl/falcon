/*********************************************************************
 * FALCON - The Falcon Programming Language.
 * FILE: list_mod.cpp
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

#define SRC "modules/native/feathers/containers/list_mod.h"

#include "list_mod.h"
#include "errors.h"
#include "iterator_mod.h"
#include <falcon/item.h>

namespace Falcon {
namespace Mod{
namespace {

class Element
{
public:
   Falcon::Item m_data;
   Element* m_next;
   Element* m_prev;
};

}

class List::Private
{
public:
   Element* m_head;
   Element* m_tail;

   Element* m_pool;
   uint32 m_poolSize;
   uint32 m_maxPoolSize;
   Mutex m_mtxPool;

   Private( int32 maxPoolSize ):
      m_head(0),
      m_tail(0),
      m_pool(0),
      m_poolSize(0),
      m_maxPoolSize(maxPoolSize)
   {}

   ~Private() {
      clearPool();
   }

   void clearPool()
   {
      m_mtxPool.lock();
      Element* elem = m_pool;
      m_poolSize = 0;
      m_pool = 0;
      m_mtxPool.unlock();

      while( elem != 0)
      {
         Element* e = elem;
         elem = elem->m_next;
         delete e;
      }
   }

   void dispose( Element* e )
   {
      m_mtxPool.lock();
      if( m_poolSize >= m_maxPoolSize )
      {
         m_mtxPool.unlock();
         delete e;
      }
      else {
         e->m_next = m_pool;
         m_pool = e;
         m_poolSize++;
         m_mtxPool.unlock();
      }
   }

   Element* alloc()
   {
      Element* ret;
      m_mtxPool.lock();
      if( m_poolSize > 0 )
      {
         ret = m_pool;
         m_pool = m_pool->m_next;
         m_poolSize--;
         m_mtxPool.unlock();

      }
      else {
         m_mtxPool.unlock();
         ret = new Element;
      }
      return ret;
   }

private:
};



namespace {


class ListIterator: public Iterator
{
public:
   int32 m_version;
   Element* m_current;

   ListIterator( Container* c, bool rev = false ):
      Iterator(c, rev),
      m_version(0),
      m_current(0)
   {
   }

   ListIterator( const ListIterator& other ):
      Iterator( other ),
      m_version( other.m_version ),
      m_current( other.m_current )
   {}

   virtual ~ListIterator(){}

   virtual bool current(Item& value) {
      if( m_current != 0 )
      {
         value = m_current->m_data;
         return true;
      }

      return false;
   }

   virtual void reset()
   {
      m_current = 0;
      m_version = 0; // overkill
   }
};


class ListFIterator: public ListIterator
{
public:
   ListFIterator( Container* c ):
      ListIterator(c)
   {}

   ListFIterator( const ListFIterator& other ):
      ListIterator( other )
   {}

   virtual ~ListFIterator(){}

   virtual bool next( Item& v )
   {
      List* list = static_cast<List*>(container());
      list->lock();
      if ( m_current == 0 )
      {
         m_current = list->prv()->m_head;
         m_version = list->version();

      }
      else {
         if( list->version() != m_version )
         {
            list->unlock();
            throw FALCON_SIGN_XERROR( ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during next") );
         }
         m_current = m_current->m_next;
      }

      if( m_current == 0 )
      {
         list->unlock();
         return false;
      }
      v = m_current->m_data;
      list->unlock();
      return true;
   }

   virtual bool hasNext()
   {
      bool value;

      List* list = static_cast<List*>(container());
      list->lock();
      if( m_current == 0 )
      {
         value = list->prv()->m_head != 0;
      }
      else {
         if( list->version() != m_version )
         {
            list->unlock();
            throw FALCON_SIGN_XERROR( ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during next") );
         }

         value = m_current->m_next != 0;
      }
      list->unlock();

      return value;
   }

   virtual Iterator* clone() const
   {
      return new ListFIterator(*this);
   }
};



class ListRIterator: public ListIterator
{
public:

   ListRIterator( Container* c ):
      ListIterator(c, true)
   {
   }

   ListRIterator( const ListIterator& other ):
      ListIterator( other )
   {}

   virtual ~ListRIterator(){}

   virtual bool next(Item& value)
   {
      List* list = static_cast<List*>(container());
      list->lock();
      if ( m_current == 0 )
      {
         m_current = list->prv()->m_tail;
         m_version = list->version();
      }
      else {
         if( list->version() != m_version )
         {
            list->unlock();
            throw FALCON_SIGN_XERROR( ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during next") );
         }
         m_current = m_current->m_prev;
      }

      if( m_current == 0 )
      {
         list->unlock();
         return false;
      }

      value = m_current->m_data;
      list->unlock();
      return true;
   }

   virtual bool hasNext()
   {
      bool value;

      List* list = static_cast<List*>(container());
      list->lock();
      if( m_current == 0 )
      {
         value = list->prv()->m_tail != 0;
      }
      else {
         if( list->version() != m_version )
         {
            list->unlock();
            throw FALCON_SIGN_XERROR( ContainerError, FALCON_ERROR_CONTAINERS_OUTSYNC, .extra("during next") );
         }

         value = m_current->m_prev != 0;
      }
      list->unlock();

      return value;
   }

   virtual Iterator* clone() const
   {
      return new ListRIterator(*this);
   }
};

}



List::List(const ClassContainerBase* cls, uint32 poolSize):
      Container(cls),
      m_size(0),
      _p(new Private(poolSize) )
{
}

List::List(const List& l):
      Container(l),
      m_size(0),
      _p(new Private(l._p->m_poolSize))
{
   l.lock();
   m_size = l.m_size;
   Element* elem = l._p->m_head;
   while( elem != 0 )
   {
      Element* copy = new Element;
      copy->m_data.copy( elem->m_data );
      copy->m_prev = _p->m_tail;
      if( _p->m_head == 0 )
      {
         _p->m_head = copy;
      }
      copy->m_next = 0;
      _p->m_tail = copy;

      elem = elem->m_next;
   }
   l.unlock();
}


List::~List()
{
   delete _p;
}

bool List::empty() const
{
   return m_size == 0;
}

int64 List::size() const
{
   return m_size;
}

Iterator* List::iterator()
{
   return new ListFIterator(this);
}

Iterator* List::riterator()
{
   return new ListRIterator(this);
}

Container* List::clone() const
{
   return new List(*this);
}

void List::append(const Item& value )
{
   push(value);
}

void List::clear()
{
   lock();
   Element* head = _p->m_head;
   _p->m_head = 0;
   _p->m_tail = 0;
   m_size = 0;
   ++m_version;
   unlock();

   while( head != 0 )
   {
      Element* h = head;
      head = head->m_next;
      delete h;
   }

}


void List::poolSize( uint32 size )
{
   _p->m_mtxPool.lock();
   _p->m_maxPoolSize = size;
   if( size == 0 )
   {
      _p->clearPool();
   }
   _p->m_mtxPool.unlock();
}


uint32 List::poolSize()
{
   _p->m_mtxPool.lock();
   uint32 size = _p->m_maxPoolSize;
   _p->m_mtxPool.unlock();
   return size;
}


bool List::erase(Iterator* iter)
{
   if( iter->container() != this )
   {
      throw FALCON_SIGN_ERROR( ContainerError, FALCON_ERROR_CONTAINERS_INVITER );
   }

   lock();
   ListIterator* i = static_cast<ListIterator*>(iter);
   if( i->m_current == 0 || _p->m_head == 0 )
   {
      unlock();
      return false;
   }

   Element* item = i->m_current;

   if( item == _p->m_head )
   {
      _p->m_head = item->m_next;
   }
   else {
      fassert( item->m_prev != 0 ); // if we're not head...
      item->m_prev->m_next= item->m_next;
   }

   if( item == _p->m_tail )
   {
      _p->m_tail = item->m_prev;
   }
   else {
      fassert( item->m_next != 0 ); // if we're not tail...
      item->m_next->m_prev = item->m_prev;
   }

   // move to the previous element.
   if( i->isReverse() )
   {
      i->m_current = item->m_next;
   }
   else
   {
      i->m_current = item->m_prev;
   }

   _p->dispose( item );
   i->m_version = ++m_version;
   unlock();

   return true;
}

bool List::insert(Iterator* iter, const Item& value )
{
   if( iter->container() != this )
   {
      throw FALCON_SIGN_ERROR( ContainerError, FALCON_ERROR_CONTAINERS_INVITER );
   }

   Element* newItem = _p->alloc();

   lock();
   ListIterator* i = static_cast<ListIterator*>(iter);
   // insert before head/after tail?
   if( i->m_current == 0 )
   {
      if( _p->m_head == 0 )
      {
         _p->m_head = _p->m_tail = newItem;
         newItem->m_next = 0;
         newItem->m_prev = 0;
         _p->m_head->m_data = value;
      }
      else {
         if( ! iter->isReverse() )
         {
            _p->m_head->m_prev = newItem;
            newItem->m_prev = 0;
            newItem->m_next = _p->m_head;
            _p->m_head = newItem;
         }
         else
         {
            _p->m_tail->m_next = newItem;
            newItem->m_next = 0;
            newItem->m_prev = _p->m_tail;
            _p->m_tail = newItem;
         }
      }
   }
   else
   {
      if( ! iter->isReverse() )
      {
         newItem->m_prev = i->m_current;
         newItem->m_next = i->m_current->m_next;
         i->m_current->m_next = newItem;
         if( newItem->m_next == 0 )
         {
            _p->m_tail = newItem;
         }
      }
      else
      {
         newItem->m_next = i->m_current;
         newItem->m_prev = i->m_current->m_prev;
         i->m_current->m_prev = newItem;
         if( newItem->m_prev == 0 )
         {
            _p->m_head = newItem;
         }
      }
   }
   unlock();

   return true;
}


void List::push(const Item& item)
{
   Element* element = _p->alloc();
   element->m_data = item;
   lock();
   element->m_prev = _p->m_tail;
   element->m_next = 0;
   if( _p->m_head == 0 )
   {
      _p->m_head = element;
   }
   else {
      _p->m_tail->m_next = element;
   }
   _p->m_tail = element;
   ++m_size;
   ++m_version;
   unlock();
}

void List::pop()
{
   lock();
   if( _p->m_tail != 0)
   {
      Element* tail = _p->m_tail;
      _p->m_tail = tail->m_prev;
      if( _p->m_tail == 0 )
      {
         _p->m_head = 0;
         m_size =0;
      }
      else {
         _p->m_tail->m_next = 0;
         --m_size;
      }
      ++m_version;
      unlock();

      _p->dispose(tail);
   }
   else
   {
      unlock();
   }
}

void List::shift()
{
   lock();
   if( _p->m_head != 0)
   {
      Element* head = _p->m_head;
      _p->m_head = head->m_next;
      if( _p->m_head == 0 )
      {
         _p->m_tail = 0;
         m_size =0;
      }
      else {
         _p->m_head->m_prev = 0;
         --m_size;
      }
      ++m_version;
      unlock();

      _p->dispose(head);
   }
   else
   {
      unlock();
   }
}

void List::unshift(const Item& item)
{
   Element* element = _p->alloc();
   element->m_data = item;
   lock();
   element->m_next = _p->m_head;
   element->m_prev = 0;
   if( _p->m_tail == 0 )
   {
      _p->m_tail = element;
   }
   else {
      _p->m_head->m_prev = element;
   }
   _p->m_head = element;
   ++m_size;
   ++m_version;
   unlock();
}

bool List::front( Item& target ) const
{
   lock();
   Element* elem = _p->m_head;
   if( elem != 0 )
   {
      target = elem->m_data;
   }
   unlock();
   return elem != 0;
}

bool List::back( Item& target ) const
{
   lock();
   Element* elem = _p->m_tail;
   if( elem != 0 )
   {
      target = elem->m_data;
   }
   unlock();
   return elem != 0;
}
}
}

/* end of list_mod.cpp */

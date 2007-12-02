/*
   FALCON - The Falcon Programming Language.
   FILE: attribute.cpp

   Standard VM attriubte item.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/attribute.h>
#include <falcon/cobject.h>

namespace Falcon
{

//=================================================================
// Attribute implementation
//=================================================================

Attribute::Attribute( const Symbol *sym ):
   m_symbol( sym ),
   m_head( 0 ),
   m_iterHead( 0 ),
   m_size( 0 )
{
}

Attribute::~Attribute()
{
   // attributes should never be destroyed while holding iterators or handlers, as they are
   // being destroyed by the VM at garbage destruction.
   // however..

   while( m_iterHead )
   {
      // this will force the iterators to change head.
      m_iterHead->invalidate();
   }

   removeFromAll();
}

void Attribute::removeFromAll()
{
   while ( m_head != 0 )
   {
      // this will also force head to move forward
      removeFrom( m_head->object() );
   }
}

bool Attribute::giveTo( CoreObject *tgt )
{
   // first, create a handler that will be stored in the object
   AttribObjectHandler *handler = new AttribObjectHandler( tgt );

   // then see if we can store the handler in the object
   AttribHandler *head = tgt->attributes();
   if ( head == 0 )
   {
      // the object had no attribute, so we are sure we can do this
      tgt->m_attributes = new AttribHandler( this, handler );
   }
   else
   {

      while( head->next() != 0 )
      {
         if ( head->attrib() == this )
         {
            // arrg, we were already given.
            delete handler;
            return false;
         }

         head = head->next();
      }

      // we have the pointer to the last attribute of the object in head
      head->next( new AttribHandler( this, handler, head, 0 ) );
   }

   // add the object to ourselves.
   if ( m_head == 0 )
      m_head = handler;
   else {
      m_head->prev( handler );
      handler->next( m_head );
      m_head = m_head->prev();
   }

   return true;
}


bool Attribute::removeFrom( CoreObject *tgt )
{
   AttribHandler *head = tgt->attributes();
   if( head == 0 )
      return false;

   // is the first element ourself ?
   if ( head->attrib() == this )
   {
      tgt->m_attributes = head->next();
      if ( tgt->m_attributes != 0 )
         tgt->m_attributes->prev(0);

      removeObject( head->objHandler() );
      delete head;
      return true;
   }

   // no, search in the rest of the list
   head = head->next();
   while ( head != 0 )
   {
      if( head->attrib() == this )
      {
         // found
         // head->prev must exist, as we have specially treated the head
         head->prev()->next( head->next() );
         // but head->next may not exist
         if( head->next() != 0 )
         {
            head->next()->prev( ( head->prev() ) );
         }

         removeObject( head->objHandler() );
         delete head;
         return true;
      }

      head = head->next();
   }

   // not found
   return false;
}

void Attribute::removeObject( AttribObjectHandler *h )
{
   if ( m_head == 0 )
   {
      delete h;
      return;
   }

   if ( h == m_head )
   {
      AttribObjectHandler *next = m_head->next();
      m_head = next;
      if ( m_head != 0 )
         m_head->prev( 0 );
   }
   else
   {
      h->prev()->next( h->next() );
      if ( h->next() != 0 )
         h->next()->prev( h->prev() );
   }

   // notify also all the iterators about this fact.
   AttribIterator *hi = m_iterHead;
   while( hi != 0 )
   {
      AttribIterator *next = hi->nextIter();
      // h should be already deleted, but we use it just for value now,
      // so it should be ok.
      hi->notifyDeletion( h );
      hi = next;
   }

   delete h;
}


AttribIterator *Attribute::getIterator()
{
   return new AttribIterator( this );
}

//=================================================================
// Attribute Iterator implementation
//=================================================================

AttribIterator::AttribIterator( Attribute *attrib )
{
   m_attrib = attrib;
   m_prev = 0;
   m_next = attrib->m_iterHead;
   if ( attrib->m_iterHead != 0 )
      m_attrib->m_iterHead->m_prev = this;

   m_attrib->m_iterHead = this;
   m_current = m_attrib->m_head;
}


AttribIterator::~AttribIterator()
{
   invalidate();
}


bool AttribIterator::next()
{
   if ( m_current == 0 )
      return false;

   m_current = m_current->next();
   return true;
}


bool AttribIterator::prev()
{
   if ( m_current == 0 )
      return false;

   m_current = m_current->prev();
   return true;
}

bool AttribIterator::hasNext() const
{
   if ( m_current == 0 || m_current->next() == 0 )
      return false;

   return true;
}


bool AttribIterator::hasPrev() const
{
   if ( m_current == 0 || m_current->prev() == 0 )
      return false;

   return true;
}


Item &AttribIterator::getCurrent() const
{
   if ( m_current )
      m_item.setObject( m_current->object() );

   return m_item;
}


bool AttribIterator::isValid() const
{
   return m_current != 0;
}

bool AttribIterator::isOwner( void *collection ) const
{
   return collection == m_current;
}


bool AttribIterator::equal( const CoreIterator &other ) const
{
   return other.isValid() && other.isOwner( m_current );
}


void AttribIterator::invalidate()
{

   if ( m_attrib != 0 )
   {
      if( m_next != 0 )
         m_next->m_prev = m_prev;

      if ( this == m_attrib->m_iterHead )
         m_attrib->m_iterHead = m_next;
      else
         m_prev->m_next = m_next;
   }

   m_current = 0;
   m_attrib = 0;
}

void AttribIterator::notifyDeletion( AttribObjectHandler *deleted )
{
   if ( deleted == m_current )
   {
      invalidate();
   }
}


UserData *AttribIterator::clone()
{
   if ( m_attrib == 0 )
   {
      return 0;
   }

   AttribIterator *other = m_attrib->getIterator();
   other->m_current = m_current;
   return other;
}

bool AttribIterator::erase()
{
   if( m_current != 0 )
   {
      AttribObjectHandler *toBeRemoved = m_current;
      m_current = m_current->next();
      m_attrib->removeFrom( toBeRemoved->object() );
      return true;
   }

   return false;
}

bool AttribIterator::insert( const Item &item )
{
   return false;
}

};

/* end of attribute.cpp */

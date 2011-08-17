/*
   FALCON - The Falcon Programming Language.
   FILE: itemlist.h

   List of Falcon Items
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   List of Falcon Items definition
*/

#ifndef flc_itemlist_H
#define flc_itemlist_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/falcondata.h>
#include <falcon/sequence.h>
#include <falcon/item.h>
#include <falcon/iterator.h>
#include <falcon/mt.h>

namespace Falcon {

class ItemListElement;
class ItemList;
class Iterator;

/** Element of a standard list of Falcon items. */
class FALCON_DYN_CLASS ItemListElement: public BaseAlloc
{
   Item m_item;

   ItemListElement *m_next;
   ItemListElement *m_prev;


public:

   /** Create the element by copying an item.
      The item is shallow copied.
   */
   ItemListElement( const Item &itm, ItemListElement *p = 0, ItemListElement *n = 0 ):
      m_item( itm ),
      m_next( n ),
      m_prev( p )
   {}

   /** Deletes the element.
        Called when all the iterators pointing to this element are gone.
    */
  ~ItemListElement()
  {
  }

   const Item &item() const { return m_item; }
   Item &item() { return m_item; }

   void next( ItemListElement *n ) { m_next = n; }
   ItemListElement *next() const { return m_next; }

   void prev( ItemListElement *p ) { m_prev = p; }
   ItemListElement *prev() const { return m_prev; }
};


/** List of Falcon items.
   This class is designed to work together with Falcon object
   as a UserData, but it can be also used for other reasons,
   when an Array is not the best way to represent data.
*/

class FALCON_DYN_CLASS ItemList: public Sequence
{
private:
   uint32 m_size;
   ItemListElement *m_head;
   ItemListElement *m_tail;

   // temporary variable using during iter-erase
   Iterator* m_erasingIter;
   ItemListElement* m_disposingElem;

public:
   /** Builds an empty list. */
   ItemList():
      m_size(0),
      m_head(0),
      m_tail(0),
      m_erasingIter(0),
      m_disposingElem(0)
   {}

   /** Clones a list. */
   ItemList( const ItemList &l );

   virtual ~ItemList()
   {
      clear();
   }

   /** Deletes the list.
      Items are shallowly destroyed.
   */
   virtual ItemList *clone() const;

   /** Gets the first item in the list.
      If the list is empty, you will crash, so use this only when the list is
      NOT empty.
      \return a reference to the first item in the list or a spectacular crash.
   */
   virtual const Item &front() const;

   /** Gets the last item in the list.
      If the list is empty, you will crash, so use this only when the list is
      NOT empty.
      \return a reference to the last item in the list or a spectacular crash.
   */
   virtual const Item &back() const;

   /** Gets the pointer to the first element for list traversal.
      The list element is just an item with previous and next pointers.
      If the list is empty, this method will return 0.
      \return the pointer to the first element pointer, or 0.
   */
   ItemListElement *first() const;

   /** Gets the pointer to the last element for list traversal.
      The list element is just an item with previous and next pointers.
      If the list is empty, this method will return 0.
      \return the pointer to the last element pointer, or 0.
   */
   ItemListElement *last() const;

   virtual void append( const Item& itm ) { push_back( itm ); }
   virtual void prepend( const Item& itm ) { push_front( itm ); }

   /** Pushes a shallow copy of the item to the end of the list.
      \param itm the item to be pushed.
   */
   void push_back( const Item &itm );

   /** Removes the last element from the list.
      The item is shallowly removed. Deep content will be reclaimed through GC.
      Calling pop_back() on an empty list will have no effect.
   */
   void pop_back();

   /** Pushes a shallow copy of the item in front of the list.
      \param itm the item to be pushed.
   */
   void push_front( const Item &itm );

   /** Removes the first element from the list.
      The item is shallowly removed. Deep content will be reclaimed by GC.
      Calling pop_front() on an empty list will have no effect.
   */
   void pop_front();

   /** Removes all the elements in the list. */
   virtual void clear();

   /** Remove given element.
      If this is the last element of the list, the method returns 0,
      else it return the element that was following the delete element
      in the list, and that now has its place.
      \param elem an element from this list (or you'll witness psychedelic crashes)
   */
   ItemListElement *erase( ItemListElement *elem );


   /** Insert an item after given before given element.
      To insert an item past the last element, use 0 as element pointer (last->next);
      this will work also to insert an item in an empty list.

      \param elem the element before which to insert the item, or 0 to apped at tail.
      \param item the item to be inserted.
   */
   void insert( ItemListElement *elem, const Item &item );


   /** Tells if the list is empty.
      \return true if the list is empty.
   */
   virtual bool empty() const { return m_size == 0; }

   /** Return the number of the items in the list.
      \return count of items in the list
   */
   uint32 size() const { return m_size; }

   /** Perform marking of items stored in the list.
   */
   virtual void gcMark( uint32 mark );

   // Deletion criterion.
   virtual bool onCriterion( Iterator* elem ) const;

   //========================================================
   // Iterator implementation.
   //========================================================
protected:

   virtual void getIterator( Iterator& tgt, bool tail = false ) const;
   virtual void copyIterator( Iterator& tgt, const Iterator& source ) const;

   virtual void insert( Iterator &iter, const Item &data );
   virtual void erase( Iterator &iter );
   virtual bool hasNext( const Iterator &iter ) const;
   virtual bool hasPrev( const Iterator &iter ) const;
   virtual bool hasCurrent( const Iterator &iter ) const;
   virtual bool next( Iterator &iter ) const;
   virtual bool prev( Iterator &iter ) const;
   virtual Item& getCurrent( const Iterator &iter );
   virtual Item& getCurrentKey( const Iterator &iter );
   virtual bool equalIterator( const Iterator &first, const Iterator &second ) const;
};


}

#endif

/* end of itemlist.h */

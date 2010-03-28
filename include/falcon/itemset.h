/*
   FALCON - The Falcon Programming Language.
   FILE: itemset.h

   (Ordered) set of falcon items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 07 Aug 2009 18:36:22 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Setof Falcon Items definition
*/

#ifndef FALCON_ITEMSET_H
#define FALCON_ITEMSET_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/falcondata.h>
#include <falcon/sequence.h>
#include <falcon/item.h>
#include <falcon/iterator.h>
#include <falcon/mt.h>

namespace Falcon {

class ItemSetElement;
class ItemSet;

/** Element of a standard set of Falcon items. */
class FALCON_DYN_CLASS ItemSetElement: public BaseAlloc
{
   Item m_item;

   ItemSetElement *m_left;
   ItemSetElement *m_right;
   ItemSetElement *m_parent;
public:

   /** Create the element by copying an item.
      The item is shallow copied.
   */
   ItemSetElement( const Item &itm, ItemSetElement* p=0, ItemSetElement *l = 0, ItemSetElement *r = 0 ):
      m_item( itm ),
      m_left( l ),
      m_right( r ),
      m_parent( p )
   {}

   /** Deletes the element.
    */
  ~ItemSetElement()
  {
  }

   const Item &item() const { return m_item; }
   Item &item() { return m_item; }

   void left( ItemSetElement *n ) { m_left = n; }
   ItemSetElement *left() const { return m_left; }

   void right( ItemSetElement *p ) { m_right = p; }
   ItemSetElement *right() const { return m_right; }

   void parent( ItemSetElement *p ) { m_parent = p; }
   ItemSetElement *parent() const { return m_parent; }
};


/** Set of Falcon items.

   This class is designed to work together with Falcon object
   as a UserData, but it can be also used alone to store unique
   entities of items.

   The set is internally represented as a binary tree (eventually
   balanced).
*/

class FALCON_DYN_CLASS ItemSet: public Sequence
{
private:
   uint32 m_size;
   ItemSetElement *m_root;
   uint32 m_mark;

   // temporary variable using during iter-erase
   Iterator* m_erasingIter;
   ItemSetElement* m_disposingElem;

   static ItemSetElement* duplicateSubTree( ItemSetElement* parent, const ItemSetElement* source );
   static void clearSubTree( ItemSetElement* source );
   static ItemSetElement* smallestInTree( ItemSetElement* e );
   static ItemSetElement* largestInTree( ItemSetElement* e );
   static bool insertInSubtree( ItemSetElement* elem, const Item& item );
   static void markSubTree( ItemSetElement* e );
   static ItemSetElement* nextElem( ItemSetElement* e );
   static ItemSetElement* prevElem( ItemSetElement* e );
   static ItemSetElement* findInTree( ItemSetElement* elem, const Item &item );

public:
   /** Builds an empty list. */
   ItemSet():
      m_size(0),
      m_root(0),
      m_mark( 0xFFFFFFFF ),
      m_erasingIter(0),
      m_disposingElem(0)
   {}

   /** Clones a list. */
   ItemSet( const ItemSet &l );

   virtual ~ItemSet()
   {
      clear();
   }

   /** Deletes the list.
      Items are shallowly destroyed.
   */
   virtual ItemSet *clone() const;

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
   ItemSetElement *first() const;

   /** Gets the pointer to the last element for list traversal.
      The list element is just an item with previous and next pointers.
      If the list is empty, this method will return 0.
      \return the pointer to the last element pointer, or 0.
   */
   ItemSetElement *last() const;

   virtual void append( const Item& itm ) { insert( itm ); }
   virtual void prepend( const Item& itm ) { insert( itm ); }


   /** Removes all the elements in the list. */
   virtual void clear();

   /** Remove given element.
      If this is the last element of the list, the method returns 0,
      else it return the element that was following the delete element
      in the list, and that now has its place.
      \param elem an element from this list (or you'll witness psychedelic crashes)
   */
   void erase( ItemSetElement *elem );

   /** Finds an item and eventually returns the relative element.
    * This function is useful for direct deletion of an item,
    * or creation of an iterator at a given position.
    */
   ItemSetElement* find( const Item &item );

   /** Creates an iterator.
    *
    */
   void getIteratorAt( Iterator &tgt, ItemSetElement* elem );


   /** Insert an item after given before given element.
      To insert an item past the last element, use 0 as element pointer (last->next);
      this will work also to insert an item in an empty list.

      \param item the item to be inserted.
   */
   void insert( const Item &item );


   /** Tells if the list is empty.
      \return true if the list is empty.
   */
   virtual bool empty() const { return m_root == 0; }

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

/* end of itemset.h */

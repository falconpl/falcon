/*
   FALCON - The Falcon Programming Language.
   FILE: itemset.cpp

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

#ifndef FALCON_itemset_H
#define FALCON_itemset_H

#include <falcon/itemset.h>

namespace Falcon {

ItemSet::ItemSet( const ItemList &l )
{

}

FalconData *ItemSet::clone() const
{
   return new ItemSet(*this);
}


const Item &ItemSet::front() const
{
   ItemSetElement* root = m_root;

   fassert( root != 0 );

   while( root->left() != 0 )
   {
      root = root->left();
   }

   return root->item();
}

const Item &ItemSet::back() const
{
   ItemSetElement* root = m_root;

   fassert( root != 0 );

   while( root->right() != 0 )
   {
      root = root->right();
   }

   return root->item();
}


ItemSetElement *ItemSet::first() const
{
   ItemSetElement* root = m_root;

   fassert( root != 0 );

   while( root->left() != 0 )
   {
      root = root->left();
   }

   return root;
}


ItemSetElement *last() const
{
   ItemSetElement* root = m_root;

   fassert( root != 0 );

   while( root->left() != 0 )
   {
      root = root->left();
   }

   return root->item();
}

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
   ItemListElement *erase( ItemSetElement *elem );


   /** Insert an item after given before given element.
      To insert an item past the last element, use 0 as element pointer (last->next);
      this will work also to insert an item in an empty list.

      \param item the item to be inserted.
   */
   void insert( const Item &item );


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

/* end of itemset.cpp */

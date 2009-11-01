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


#include <falcon/itemset.h>
#include <falcon/mempool.h>
#include <falcon/error.h>
#include <falcon/eng_messages.h>

namespace Falcon {

ItemSet::ItemSet( const ItemSet &l )
{
   m_root = duplicateSubTree( 0, l.m_root );
   m_size = l.m_size;
}


ItemSetElement* ItemSet::duplicateSubTree( ItemSetElement* parent, const ItemSetElement* source )
{
   if ( source == 0 )
      return 0;

   ItemSetElement* ret = new ItemSetElement( source->item(), parent );
   ret->left( duplicateSubTree( ret, source->left() ) );
   ret->right( duplicateSubTree( ret, source->right() ) );

   return ret;
}


ItemSet *ItemSet::clone() const
{
   return new ItemSet(*this);
}


const Item &ItemSet::front() const
{
   fassert( ! empty() );
   return first()->item();
}


const Item &ItemSet::back() const
{
   fassert( ! empty() );
   return last()->item();
}


ItemSetElement *ItemSet::first() const
{
   if ( m_root == 0 )
      return 0;

   return smallestInTree( m_root );
}

ItemSetElement *ItemSet::smallestInTree( ItemSetElement *e )
{
   while( e->left() != 0 )
   {
      e = e->left();
   }

   return e;
}


ItemSetElement *ItemSet::last() const
{
   if ( m_root == 0 )
      return 0;

   return largestInTree( m_root );
}

ItemSetElement *ItemSet::largestInTree( ItemSetElement *e )
{
   while( e->right() != 0 )
   {
      e = e->right();
   }

   return e;
}


void ItemSet::clear()
{
   invalidateAllIters();
   clearSubTree( m_root );
   m_size = 0;
}

void ItemSet::clearSubTree( ItemSetElement *e )
{
   if( e != 0 )
   {
      clearSubTree( e->left() );
      clearSubTree( e->right() );
      delete e;
   }
}

void ItemSet::erase( ItemSetElement *elem )
{
   // the new subtree root is the highest element in the left subtree.
   ItemSetElement *newRoot = 0;

   if ( elem->left() )
   {
      newRoot = largestInTree( elem->left() );
      newRoot->parent()->right( newRoot->left() );
   }
   else if( elem->right() ) {
      newRoot = smallestInTree( elem->right() );
      newRoot->parent()->left( newRoot->right() );
   }

   // was this element the main root?
   if ( elem->parent() == 0 )
   {
      m_root = newRoot;
   }
   else
   {
      // disengage the element from the parent

      if( elem->parent()->left() == elem )
         elem->parent()->left( newRoot );
      else
         elem->parent()->right( newRoot );

   }

   // assign the new children
   newRoot->left( elem->left() );
   newRoot->right( elem->right() );

   // the element is disengaged.
   // invalidate the iterators pointing here.
   m_disposingElem = elem;
   invalidateIteratorOnCriterion();
   m_disposingElem = 0;

   // time to get rid of the element.
   delete elem;
   --m_size;
}


ItemSetElement* ItemSet::find( const Item &item )
{
   if( m_root == 0 )
      return 0;
   return findInTree( m_root, item );
}


ItemSetElement* ItemSet::findInTree( ItemSetElement* elem, const Item &item )
{
   int c = elem->item().compare( item );
   if ( c < 0 )
   {
      if ( elem->right() != 0 )
         return findInTree( elem->right(), item );
   }
   else if ( c > 0 )
   {
      if ( elem->left() != 0 )
         return findInTree( elem->left(), item );
   }
   else // ==
      return elem;

   return 0;  // not found,
}

void ItemSet::getIteratorAt( Iterator &tgt, ItemSetElement* elem )
{
   Sequence::getIterator( tgt, false );
   tgt.data( elem );
}

void ItemSet::insert( const Item &item )
{
   if( m_root == 0 )
   {
      m_root = new ItemSetElement( item );
      ++m_size;
   }
   else
   {
      if( insertInSubtree( m_root, item ) )
         ++m_size;
   }
}


bool ItemSet::insertInSubtree( ItemSetElement* elem, const Item& item )
{
   int result = elem->item().compare( item );
   if( result == 0 )
   {
      elem->item() = item;
      return false;
   }

   if( result < 0 )
   {
      if ( elem->right() != 0 )
         return insertInSubtree( elem->right(), item );
      else
         elem->right( new ItemSetElement( item, elem ) );
   }
   else
   {
      if ( elem->left() != 0 )
        return insertInSubtree( elem->left(), item );
     else
        elem->left( new ItemSetElement( item, elem ) );
   }

   // if we arrived here, it means we added a new node.
   return true;
}


ItemSetElement* ItemSet::nextElem( ItemSetElement* e )
{
   if( e->right() != 0 )
      return smallestInTree( e->right() );

   while( e->parent() != 0 && e->parent()->right() == e )
      e = e->parent();

   return e->parent();
}


ItemSetElement* ItemSet::prevElem( ItemSetElement* e )
{
   if( e->left() != 0 )
      return largestInTree( e->left() );

   while( e->parent() != 0 && e->parent()->left() == e )
      e = e->parent();

   return e->parent();
}


void ItemSet::gcMark( uint32 m )
{
   if( m_mark != m )
   {
      m_mark = m;
      Sequence::gcMark( m );
      markSubTree( m_root );
   }
}


void ItemSet::markSubTree( ItemSetElement* e )
{
   if( e != 0 )
   {
      memPool->markItem( e->item() );
      markSubTree( e->left() );
      markSubTree( e->right() );
   }
}


bool ItemSet::onCriterion( Iterator* elem ) const
{
   return elem->data() == m_disposingElem && elem != m_erasingIter;
}


void ItemSet::getIterator( Iterator& tgt, bool tail ) const
{
   Sequence::getIterator( tgt, tail );
   if ( m_root == 0 )
      tgt.data(0);
   else if( tail )
      tgt.data( largestInTree( m_root ) );
   else
      tgt.data( smallestInTree( m_root ) );
}

void ItemSet::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   Sequence::copyIterator( tgt, source );
   tgt.data( source.data() );
}


void ItemSet::insert( Iterator &iter, const Item &data )
{
   insert( data );
}

void ItemSet::erase( Iterator &iter )
{
   m_erasingIter = &iter;
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   next( iter );
   erase( (ItemSetElement*) elem );
   m_erasingIter = 0;
}

bool ItemSet::hasNext( const Iterator &iter ) const
{
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   return elem != 0 && nextElem( elem ) != 0;
}

bool ItemSet::hasPrev( const Iterator &iter ) const
{
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   return elem == 0 || prevElem( elem ) != 0;
}

bool ItemSet::hasCurrent( const Iterator &iter ) const
{
  return iter.data() != 0;
}

bool ItemSet::next( Iterator &iter ) const
{
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   if ( elem == 0 )
      return false;

   iter.data( nextElem( elem ) );
   return iter.data() != 0;
}

bool ItemSet::prev( Iterator &iter ) const
{
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   if ( elem == 0 )
   {
      if ( m_root )
         iter.data( largestInTree( m_root ) );
      // otherwise, it stays 0
   }
   else
      iter.data( prevElem( elem ) );

   return iter.data() != 0;
}

Item& ItemSet::getCurrent( const Iterator &iter )
{
   ItemSetElement* elem = (ItemSetElement*) iter.data();
   fassert( elem != 0 );
   return elem->item();
}

Item& ItemSet::getCurrentKey( const Iterator &iter )
{
   throw new CodeError( ErrorParam( e_non_dict_seq, __LINE__ )
                 .origin( e_orig_runtime ).extra( "ItemSet::getCurrentKey" ) );
}

bool ItemSet::equalIterator( const Iterator &first, const Iterator &second ) const
{
   return first.data() == second.data();
}

}

/* end of itemset.cpp */

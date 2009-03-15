 /*
   FALCON - The Falcon Programming Language.
   FILE: genericmap.cpp

   Generic map - a map holding generic values.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun ago 23 21:55:38 CEST 2004


   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/genericmap.h>
#include <falcon/memory.h>
#include <falcon/string.h>
#include <falcon/fassert.h>

#include <string.h>

#define PLAT_ALIGN 4

namespace Falcon
{

Map::Map( ElementTraits *keyt, ElementTraits *valuet, uint16 order ):
      m_keyTraits( keyt ),
      m_valueTraits( valuet ),
      m_treeOrder( order ),
      m_size(0),
      m_treeTop(0)
{
   if ( order % 2 == 0 )
      m_treeOrder = order + 1;

   // pre-cache key and value sizes
   m_keySize = keyt->memSize();

   // force 4bytes alignment
   if ( m_keySize % PLAT_ALIGN != 0 )
   {
      m_keySize = ((m_keySize / PLAT_ALIGN ) + 1) * PLAT_ALIGN;
   }

   m_valueSize = valuet->memSize();
   // force 4bytes alignment
   if ( m_valueSize % PLAT_ALIGN != 0 )
   {
      m_valueSize = ((m_valueSize / PLAT_ALIGN) + 1) * PLAT_ALIGN;
   }

   // create the first page.
   m_treeTop = allocPage();
}

Map::~Map()
{
   if ( m_treeTop != 0 )
   {
      destroyPage( m_treeTop );
      m_treeTop = 0;
   }
   else {
      throw "Double destruction";
   }
}

//======================================
//

MAP_PAGE *Map::allocPage() const
{
   fassert( sizeof(MAP_PAGE) % 4 == 0 );

   uint32 sz = (m_keySize + m_valueSize + sizeof( MAP_PAGE * ) ) * m_treeOrder + sizeof( MAP_PAGE );

   MAP_PAGE *page = (MAP_PAGE *) memAlloc( sz );
   page->m_count = 0;
   page->m_parent = 0;
   page->m_higher = 0;
   page->m_allocated = 0;
   page->m_dummy = 0;
   page->m_parentElement = 0;

   return page;
}

MAP_PAGE **Map::ptrsOfPage( const MAP_PAGE *ptr ) const
{
   char *page = (char *) ptr;
   return (MAP_PAGE **) ( page + sizeof( MAP_PAGE ) );
}

void *Map::keysOfPage( const MAP_PAGE *ptr ) const
{
   char *page = (char *) ptr;
   return page + sizeof( MAP_PAGE ) + ( m_treeOrder * sizeof( MAP_PAGE * ) );
}

void *Map::valuesOfPage( const MAP_PAGE *ptr ) const
{
   char *page = (char *) ptr;
   return page + sizeof( MAP_PAGE ) + ( m_treeOrder * (sizeof( MAP_PAGE * ) + m_keySize ) );
}

MAP_PAGE *Map::ptrInPage( const MAP_PAGE *ptr, uint16 count ) const
{
   char *page = (char *) ptr;
   MAP_PAGE **pageVect = (MAP_PAGE **) ( page + sizeof( MAP_PAGE ) );
   return pageVect[count];
}

void *Map::keyInPage( const MAP_PAGE *ptr, uint16 count ) const
{
   char *page = (char *) ptr;
   page += sizeof( MAP_PAGE ) + ( m_treeOrder * sizeof( MAP_PAGE * ) );
   return page + (count * m_keySize);
}

void *Map::valueInPage( const MAP_PAGE *ptr, uint16 count ) const
{
   char *page = (char *) ptr;
   page += sizeof( MAP_PAGE ) + ( m_treeOrder * (sizeof( MAP_PAGE * ) + m_keySize ) );
   return page + (count * m_valueSize);
}


//======================================
//
void *Map::find( const void *key ) const
{
   MapIterator iter;
   if( find( key, iter ) )
      return iter.currentValue();
   return 0;
}

bool Map::find( const void *key, MapIterator &iter ) const
{
   iter.m_map = this;

   // if the top page has zero element, this is the find
   if( m_treeTop->m_count == 0 )
   {
      iter.m_pagePosition = 0;
      iter.m_page = m_treeTop;

      return false;
   }

   return subFind( key, iter, m_treeTop );
}


bool Map::subFind( const void *key, MapIterator &iter, MAP_PAGE *currentPage ) const
{
   register int count = currentPage->m_count;

   // By design, this page cannot have zero elements

   uint16 pos;
   bool found = scanPage( key, currentPage, count, pos );

   if ( found )
   {
      // FOUND!!!
      iter.m_pagePosition = pos;
      iter.m_page = currentPage;
      return true;
   }

   // greater than the greatest element?
   if( pos >= count )
   {
      // if there is a greater page, search there
      if ( currentPage->m_higher != 0 )
      {
         return subFind( key, iter, currentPage->m_higher );
      }

      // else, insert past last
      iter.m_pagePosition = count;
      iter.m_page = currentPage;
      return false;
   }

   // if the item has a smaller page, go there
   MAP_PAGE *page = ptrInPage( currentPage, pos );
   if ( page != 0 ) {
      return subFind( key, iter, page );
   }

   // we should insert in this position
   iter.m_pagePosition = pos;
   iter.m_page = currentPage;
   return false;
}



bool Map::scanPage( const void *key, MAP_PAGE *currentPage, uint16 higher, uint16 &ret_pos ) const
{
   // by design, higher can't be zero.
   register uint16 lower = 0, point;
   higher --;

   point = higher / 2;
   void *cfrKey;
   int cmp;

   while ( true )
   {
      // get the table element
      cfrKey = keyInPage( currentPage, point );
      cmp = m_keyTraits->compare( cfrKey, key );

      if( cmp == 0 ) {
         ret_pos = point;
         return true;
      }
      else
      {
         if ( lower == higher )  // not found
         {
            break;
         }
         // last try. In pair sized dictionaries, it can be also in the other node
         else if ( point == lower && point == higher - 1 )
         {
            // if it's lower than the lower, we must insert before the lower )
            if ( cmp > 0 )
            {
               ret_pos = point;
               return false;
            }

            // being integer math, ulPoint is rounded by defect and has
            // already looked at the ulLower position
            point = lower = higher;
            // try again
            continue;
         }

         if ( cmp < 0 )
         {
            lower = point;
         }
         else
         {
            higher = point;
         }
         point = ( lower + higher ) / 2;
      }
   }

   // entry not found, but signal the best match anyway
   ret_pos =  cmp < 0 ? higher + 1: higher;

   return false;
}

void Map::insertSpaceInPage( MAP_PAGE *page, uint16 pos )
{
   // by design, count in page cannot be greater than page order
   if( pos < page->m_count )
   {
      char *mp_pos = (char *) page;
      mp_pos += sizeof( MAP_PAGE ) + (sizeof( MAP_PAGE *) * pos);
      memmove( mp_pos + sizeof( MAP_PAGE * ), mp_pos, sizeof( MAP_PAGE *) * (page->m_count - pos ) );

      char *key_pos = (char *) keyInPage( page, pos );
      memmove( key_pos + m_keySize, key_pos, m_keySize * (page->m_count - pos ) );

      char *val_pos = (char *) valueInPage( page, pos );
      memmove( val_pos + m_valueSize, val_pos, m_valueSize * (page->m_count - pos ) );
   }
   page->m_count++;
}

void Map::removeSpaceFromPage( MAP_PAGE *page, uint16 pos )
{
   // The last element does not need refitting
   if( pos < page->m_count - 1 )
   {
      char *mp_pos = (char *) page;
      mp_pos += sizeof( MAP_PAGE ) + (sizeof( MAP_PAGE *) * pos);
      memmove( mp_pos, mp_pos + sizeof( MAP_PAGE *), sizeof( MAP_PAGE *) * (page->m_count - pos -1) );

      char *key_pos = (char *) keyInPage( page, pos );
      memmove( key_pos, key_pos + m_keySize, m_keySize * (page->m_count - pos -1) );

      char *val_pos = (char *) valueInPage( page, pos );
      memmove( val_pos, val_pos + m_valueSize, m_valueSize * (page->m_count - pos -1) );

      page->m_count --;

      if ( ptrInPage( page, pos ) != 0 )
      {
         while ( pos < page->m_count )
         {
            ptrInPage( page, pos )->m_parentElement = pos;
            ++pos;
         }
      }
   }
   else
      page->m_count --;
}

bool Map::insert( const void *key, const void *value )
{
   MapIterator iter;

   if ( find( key, iter ) )
   {
      m_valueTraits->destroy( iter.currentValue() );
      m_valueTraits->copy( iter.currentValue(), value );
      return false;
   }

   // fix page situation.
   insertSpaceInPage( iter.m_page, iter.m_pagePosition );

   // put data in space
   m_keyTraits->copy( iter.currentKey(), key );
   m_valueTraits->copy( iter.currentValue(), value );
   ptrsOfPage( iter.m_page )[ iter.m_pagePosition ] = 0;

   // as this is a leaf, no extra management is needed

   // signal we have an element more
   m_size++;

   // if the page is full, balance.
   if ( iter.m_page->m_count == m_treeOrder )
   {
      splitPage( iter.m_page );
   }

   return true;
}


bool Map::erase( const void *key )
{
   MapIterator iter;

   if ( find( key, iter ) )
   {
      erase( iter );
      return true;
   }

   return false;
}

MapIterator Map::erase( const MapIterator &iter )
{
   void *key = keyInPage( iter.m_page, iter.m_pagePosition );
   void *value = valueInPage( iter.m_page, iter.m_pagePosition );
   m_keyTraits->destroy( key );
   m_valueTraits->destroy( value );
   MAP_PAGE *child = ptrInPage( iter.m_page, iter.m_pagePosition );

	MapIterator retIter = iter;

   // if we have no children, we must shrink the page.
   if ( child == 0 )
   {
      removeSpaceFromPage( iter.m_page, iter.m_pagePosition );

      // if we are too small, we must re-balance the tree
      // but the tree-top is an exception
      if( iter.m_page->m_count < m_treeOrder / 2 && iter.m_page != m_treeTop )
      {
         rebalanceNode( iter.m_page, &retIter );
      }
   }
   else {
      // we'll promote the highest of our children to our position.
      MAP_PAGE *child_child = child->m_higher;

      // and promote one from each child up to the leaves.
      while( child_child != 0 )
      {
         child = child_child;
         child_child = child->m_higher;
      }
      child->m_count--;
      memcpy( key, keyInPage( child, child->m_count ), m_keySize );
      memcpy( value, valueInPage( child, child->m_count ), m_valueSize );

      // in case a leaf child is unbalanced, we'll start rebalance algorithm.
      if( child->m_count < m_treeOrder / 2 )
         rebalanceNode( child, &retIter );

   }

   m_size --;

   return retIter;
}



MAP_PAGE *Map::getLeftSibling( const MAP_PAGE *page ) const
{
   uint16 parentElem = page->m_parentElement;
   MAP_PAGE *parent = page->m_parent;

   // No parent, no sibling.
   if ( parent == 0 || parentElem == 0 )
      return 0;

   if ( parentElem >= parent->m_count )
      return ptrInPage( parent, parent->m_count - 1 );

   return ptrInPage( parent, parentElem - 1 );
}


MAP_PAGE *Map::getRightSibling( const MAP_PAGE *page ) const
{
   uint16 parentElem = page->m_parentElement;
   MAP_PAGE *parent = page->m_parent;

   // No parent, no sibling.
   if ( parent == 0 )
      return 0;

   // we are the higher; of course we don't have siblings.
   if ( parentElem >= parent->m_count )
      return 0;

   // ok, we have a sibling in the parent
   parentElem++;
   if( parentElem >= parent->m_count )
      return parent->m_higher;

   return ptrInPage( parent, parentElem );
}



void Map::reshapeChildPointers( MAP_PAGE *page, uint16 startFrom )
{
   while ( startFrom < page->m_count )
   {
      MAP_PAGE *child = ptrInPage( page, startFrom );
      child->m_parent = page;
      child->m_parentElement = startFrom;
      ++ startFrom;
   }
   page->m_higher->m_parentElement = m_treeOrder + 1;
   page->m_higher->m_parent = page;
}



void Map::rebalanceNode( MAP_PAGE *page, MapIterator *scanner )
{
   MAP_PAGE *left, *right;
   MAP_PAGE *parent;

   parent = page->m_parent;
   int limit = m_treeOrder / 2;
   // no rebalancing for the root
   if ( parent == 0 || page->m_count >= limit )
      return;

   // identify left sibling.
   left = getLeftSibling( page );
   right = getRightSibling( page );

   // nonroot element must have at least a sibling.
   fassert( left != 0 || right != 0 );

   // decide which has the larger count.
   if ( (right != 0 && right->m_count > limit ) && ( left == 0  || right->m_count > left->m_count ) )
   {
      // rotate right elements
      int elems = (right->m_count - limit) / 2;

      // move our parent here at limit position
      memcpy( keyInPage( page, page->m_count ), keyInPage( parent, page->m_parentElement), m_keySize );
      memcpy( valueInPage( page, page->m_count ), valueInPage( parent, page->m_parentElement), m_valueSize );

      // whose child is our higher
      ptrsOfPage(page)[ page->m_count ] = page->m_higher;

		// if the scanner was at our parent, move it to limit in this page

      // now move elems items from the page on the right.
		// elems may be zero if the other page is just limit + 1 items.
		if ( elems > 0 )
		{
			memcpy( keyInPage( page, page->m_count + 1), keyInPage( right, 0 ), m_keySize * elems );
			memcpy( valueInPage( page, page->m_count + 1 ), valueInPage( right, 0 ), m_valueSize * elems );
			memcpy( &ptrsOfPage(page)[ page->m_count + 1 ], &ptrsOfPage( right )[ 0 ] , sizeof( MAP_PAGE *) * elems );
		}

      // now rotate the elems item in place of our old parent; its' child are our new higher
      memcpy( keyInPage( parent, page->m_parentElement), keyInPage( right, elems ), m_keySize );
      memcpy( valueInPage( parent, page->m_parentElement), valueInPage( right, elems ), m_valueSize );
      page->m_higher = ptrInPage( right, elems );

      // set our new count
      page->m_count = page->m_count + elems + 1;

      // finally, shift left the right pages of elems + 1 items.
      elems ++;
      int rcount = right->m_count - elems;
      memmove( keyInPage( right, 0 ), keyInPage( right, elems ), m_keySize * rcount );
      memmove( valueInPage( right, 0), valueInPage( right, elems ), m_valueSize * rcount );
      memmove( ptrsOfPage( right ), &ptrsOfPage( right )[elems], sizeof( MAP_PAGE *) * rcount );
      right->m_count = rcount;

      // fix backpointers to changed pages.
      if ( ptrInPage( page, 0 ) != 0 )
      {
         reshapeChildPointers( page );
         reshapeChildPointers( right );
      }

		// if the scanner was in the moved elements, move it too
		if ( scanner != 0 )
		{
			if ( scanner->m_page == parent && scanner->m_pagePosition == page->m_parentElement )
			{
				scanner->m_page = page;
				scanner->m_pagePosition = limit;
			}
			else if ( scanner->m_page == right )
			{
				// elems has been grown
				if ( scanner->m_pagePosition == elems - 1)
				{
					scanner->m_page = parent;
					scanner->m_pagePosition = page->m_parentElement;
				}
				else if ( scanner->m_pagePosition < elems -1 ) {
					scanner->m_page = page;
					scanner->m_pagePosition = limit + 1 + scanner->m_pagePosition;
				}
				else {
					scanner->m_pagePosition -= elems;
				}
			}
		}

      return;
   }

   if ( left != 0 && left->m_count > limit )
   {
      // rotate left elements
      int elems = (left->m_count - limit) / 2;

      // shift this page elements on the right to make room (elems plus the rotated parent
      memmove( keyInPage( page, elems + 1 ), keyInPage( page, 0 ), m_keySize * page->m_count );
      memmove( valueInPage( page, elems + 1 ), valueInPage( page, 0 ), m_valueSize * page->m_count );
      memmove( &ptrsOfPage( page )[elems + 1], ptrsOfPage( page ), sizeof( MAP_PAGE *) * page->m_count );

      // move left page parent's to elems position.
      memcpy( keyInPage( page, elems ), keyInPage( parent, left->m_parentElement), m_keySize );
      memcpy( valueInPage( page, elems ), valueInPage( parent, left->m_parentElement), m_valueSize );

      // whose child left's higher
      ptrsOfPage(page)[elems] = left->m_higher;

      // now move elems items from the page on the left.
      int lcount = left->m_count - elems;

		// elems may be zero if the other page is just limit + 1 items.
		if ( elems > 0 )
		{
			memcpy( keyInPage( page, 0 ), keyInPage( left, lcount ), m_keySize * elems);
			memcpy( valueInPage( page, 0 ), valueInPage( left, lcount ), m_valueSize * elems );
			memcpy( ptrsOfPage(page), &ptrsOfPage( left )[lcount], sizeof( MAP_PAGE *) * elems );
		}

      // now rotate the elems item in place of left's old parent; its' child are left's higher
      lcount--;
      memcpy( keyInPage( parent, left->m_parentElement), keyInPage( left, lcount ), m_keySize );
      memcpy( valueInPage( parent, left->m_parentElement), valueInPage( left, lcount ), m_valueSize );
      left->m_higher = ptrInPage( left, lcount );

      // set our new count
      page->m_count +=  elems + 1;

      // finally, shift left the right pages of elems + 1 items.
      left->m_count = lcount;

      // fix backpointers to changed pages.
      if ( ptrInPage( page, 0 ) != 0 )
      {
         reshapeChildPointers( page );
         // only the higher is changed in the left page
         left->m_higher->m_parent = left;
         left->m_higher->m_parentElement = m_treeOrder;
      }

		// if the scanner was in the moved elements, move it too
		if ( scanner != 0 )
		{
			// if the scanner was at our parent, move it to limit in this page
			if ( scanner->m_page == parent && scanner->m_pagePosition == left->m_parentElement )
			{
				scanner->m_page = page;
				scanner->m_pagePosition = elems;
			}
			else if ( scanner->m_page == left )
			{
				// lcount has already been shrunk
				if ( scanner->m_pagePosition == lcount )
				{
					scanner->m_page = parent;
					scanner->m_pagePosition = left->m_parentElement;
				}
				else if ( scanner->m_pagePosition > lcount )
				{
					scanner->m_page = page;
					scanner->m_pagePosition = scanner->m_pagePosition - lcount - 1;
				}
			}
			else if ( scanner->m_page == page )
			{
				scanner->m_pagePosition = elems + scanner->m_pagePosition;
			}
		}

      return;
   }

	// if here, we can only perform a complete merge.

   // If right is not zero, excange us with left and right with page, and act as for left.
   if ( left == 0 )
   {
      left = page;
      page = right;
   }

   // we need a bit of space on the left.
   memcpy( keyInPage( page, left->m_count + 1), keysOfPage( page ), m_keySize * page->m_count );
   memcpy( valueInPage( page, left->m_count + 1), valuesOfPage( page ), m_valueSize * page->m_count );
   memcpy( &ptrsOfPage( page )[ left->m_count + 1], ptrsOfPage( page ), sizeof(MAP_PAGE *) * page->m_count );

   // nowy copy the keys on the left (0 to left->m_count -1)
   memcpy( keysOfPage( page ), keysOfPage( left ), m_keySize * left->m_count );
   memcpy( valuesOfPage( page ) , valuesOfPage( left ), m_valueSize * left->m_count  );
   memcpy( ptrsOfPage( page ), ptrsOfPage( left ), sizeof(MAP_PAGE *) * left->m_count );

   memcpy( keyInPage( page, left->m_count ), keyInPage( parent, left->m_parentElement ), m_keySize  );
   memcpy( valueInPage( page, left->m_count ) , valueInPage( parent, left->m_parentElement ), m_valueSize  );
   ptrsOfPage( page )[ left->m_count ]  = left->m_higher;
   page->m_count = left->m_count + page->m_count + 1; // page should be full now except for 1

   removeSpaceFromPage( parent, left->m_parentElement );

	// if the scanner was in the moved elements, move it too
	if ( scanner != 0 )
	{
		// if the scanner was at our parent, move it to limit in this page
		if ( scanner->m_page == parent )
		{
			if( scanner->m_pagePosition == left->m_parentElement )
			{
				scanner->m_page = page;
				scanner->m_pagePosition = limit;
			}
			else if ( scanner->m_pagePosition > left->m_parentElement )
			{
				scanner->m_pagePosition--;
			}
		}
		else if ( scanner->m_page == left )
		{
			scanner->m_page = page;
		}
		else if ( scanner->m_page == page )
		{
			scanner->m_pagePosition = limit + 1 + scanner->m_pagePosition;
		}
	}

   memFree( left );

   if ( ptrInPage( page, 0 ) != 0 )
      reshapeChildPointers( page );

   if ( parent->m_count < limit )
   {
      // treetop?
      if( parent == m_treeTop )
      {
         if ( parent->m_count == 0 )
         {
            // page was the higher of treetop...
            memFree( m_treeTop );
            m_treeTop = page;
            page->m_parent = 0;
         }
      }
      else
         rebalanceNode( parent, scanner );
   }


   return;
}

void Map::splitPage( MAP_PAGE *page )
{
   // splitting a page requires to insert the median element in the upper page.
   MAP_PAGE *parent = page->m_parent;

   void *key;
   void *value;
   int i;

   // this is the splitted node
   uint16 splitPos = page->m_count / 2;
   key = keyInPage( page, splitPos );
   value = valueInPage( page, splitPos );
   MAP_PAGE *selected_child = ptrInPage( page, splitPos );

   // create a new page that will be added to the left of this page
   MAP_PAGE *new_left = allocPage();
   memcpy( ptrsOfPage( new_left ), ptrsOfPage( page ), sizeof( MAP_PAGE *) * splitPos );
   memcpy( keysOfPage( new_left ), keysOfPage( page ), m_keySize * splitPos );
   memcpy( valuesOfPage( new_left ), valuesOfPage( page ), m_valueSize * splitPos );
   new_left->m_count = splitPos;

   // if we don't have a parent (if we are the treetop), we must create a new treetop
   if( parent == 0 )
   {
      fassert( page == m_treeTop );

      // we must create a new treetop whose higher pointer is the splitted page.
      parent = allocPage();
      parent->m_parent = 0;
      memcpy( keysOfPage( parent ), key , m_keySize );
      memcpy( valuesOfPage( parent ), value, m_valueSize );

      ptrsOfPage( parent ) [ 0 ] = new_left;
      parent->m_higher = page;
      parent->m_count = 1;

      new_left->m_parent = parent;
      new_left->m_parentElement = 0;

      page->m_parent = parent;
      page->m_parentElement = m_treeOrder;

      m_treeTop = parent;
   }
   else {
      fassert( page != m_treeTop );

      // Now save the element in the previous page
      // place the inserted item
      uint16 parentPos = page->m_parentElement;

      // insert the splitted item
      if( parentPos < parent->m_count )
         insertSpaceInPage( parent, parentPos );
      else {
         parentPos = parent->m_count;
         parent->m_count ++;
      }

      memcpy( keyInPage( parent, parentPos ), key , m_keySize );
      memcpy( valueInPage( parent, parentPos ), value, m_valueSize );
      // the page maintain the same parent, that is moved forward by one
      for( uint16 childPos = parentPos + 1; childPos < parent->m_count; childPos ++  )
      {
         ptrInPage( parent, childPos )->m_parentElement = childPos;
      }

      // the child of the inserted splitted element is the new left page
      ptrsOfPage( parent ) [ parentPos ] = new_left;
      new_left->m_parent = parent;
      new_left->m_parentElement = parentPos;
   }

   // the old child of the splitted element becomes the new higher of the left page
   if ( selected_child != 0 )
   {
      selected_child->m_parent = new_left;
      selected_child->m_parentElement = m_treeOrder;
      new_left->m_higher = selected_child;
   }
   else
      new_left->m_higher = 0;

   // now that we're done with the key, we can scroll back the original page.
   splitPos++;
   int scrollSize = page->m_count - splitPos;
   memcpy( ptrsOfPage( page ), ptrsOfPage( page ) + splitPos, sizeof( MAP_PAGE *) * scrollSize );
   memcpy( keysOfPage( page ), keyInPage( page, splitPos ) , m_keySize * scrollSize );
   memcpy( valuesOfPage( page ), valueInPage( page, splitPos ), m_valueSize * scrollSize );
   page->m_count = scrollSize;

   // we have to update all the children page to point to the new page positions.
   MAP_PAGE *child = ptrInPage( new_left, 0 );
   if ( child != 0 )
   {
      fassert( new_left->m_count == page->m_count ); // a little check
      for( i = 0; i < page->m_count; i++ )  // we've just set it to page count
      {
         child = ptrInPage( new_left, i );
         fassert( child != 0 );
         child->m_parent = new_left;
         child->m_parentElement = i;

         child = ptrInPage( page, i );
         fassert( child != 0 );
         // parent was already page.
         child->m_parentElement = i;
      }
   }

   // higher elements have already been updated correctly.
   // the only thing left to do is to see if the parent has overgrown.
   if( parent->m_count == m_treeOrder )
   {
      splitPage( parent );
   }
}

void Map::clear()
{
   destroyPage( m_treeTop );
   m_treeTop = allocPage();
   m_size = 0;
}

void Map::destroyPage( MAP_PAGE *page )
{
   for ( uint16 i = 0; i < page->m_count; i++ )
   {
      m_keyTraits->destroy( keyInPage( page, i ) );
      m_valueTraits->destroy( valueInPage( page, i ) );
      MAP_PAGE *child = ptrInPage( page, i );
      if ( child != 0 )
         destroyPage( child );
   }

   if ( page->m_higher != 0 )
      destroyPage( page->m_higher );

   memFree( page );
}

MapIterator Map::begin() const
{
   MAP_PAGE *page = m_treeTop;
   if ( m_size == 0 ) {
      return MapIterator( this, 0, 0 );
   }
   MAP_PAGE *next = ptrInPage( page, 0 );

   while( next != 0 )
   {
      page = next;
      next = ptrInPage( page, 0 );
   }

   MapIterator iter( this, page, 0 );
   return iter;
}

MapIterator Map::end() const
{
   MAP_PAGE *page = m_treeTop;
   if ( m_size == 0 ) {
      return MapIterator( this, 0, 0 );
   }

   while( page->m_higher != 0 )
      page = page->m_higher;

   // will generate an invalid iterator. prev() must be used.
   MapIterator iter( this, page, page->m_count );
   return iter;
}

bool MapIterator::next()
{
   // if it's the same page, go to the left.
   m_pagePosition++;
   MAP_PAGE *page = m_page;

   // if the current page is over...
   if ( m_pagePosition >= page->m_count )
   {
      // if we have a higher page, use that
      if( page->m_higher != 0 )
      {
         m_page = page->m_higher;
         page = m_map->ptrInPage( m_page, 0 );
         while( page != 0 )
         {
            m_page = page;
            page = m_map->ptrInPage( page, 0 );
         }

         m_pagePosition = 0;
         return true;
      }

      // get parent's sibling.
      int16 parentPos = page->m_parentElement;
      page = page->m_parent;
      while( page != 0 )
      {
         if ( parentPos < page->m_count )
         {
            // return our parent
            m_pagePosition = parentPos;
            m_page = page;
            return true;
         }

         // we were from an higher, so we can't get again in an higher.
         parentPos = page->m_parentElement;
         page = page->m_parent;
      }

      // we're off
      return false;

   }
   else {
      // get the child of our next sibling
      page = m_map->ptrInPage( page, m_pagePosition );
      if ( page == 0 )
      {
         // return our sibling ( as we already did m_pagePosition++)
         return true;
      }
      else {
         // descend to the bottom of the hyerarcy
         m_page = page;
         page = m_map->ptrInPage( page, 0 );
         while( page != 0 )
         {
            m_page = page;
            page = m_map->ptrInPage( page, 0 );
         }

         m_pagePosition = 0;
         return true;
      }
   }

   // we never get here
   fassert( false );
}


bool MapIterator::prev()
{
   // has this element a child ? - in this case, get the leftmost child element.
   if( m_pagePosition < m_page->m_count )
   {
      MAP_PAGE *child = m_map->ptrInPage( m_page, m_pagePosition );
      if( child != 0 )
      {
         while( child->m_higher != 0 )
         {
            child = child->m_higher;
         }

         m_page = child;
         m_pagePosition = child->m_count - 1;
         return true;
      }

      // if we have no children, proceed as usual
   }

   //are there other elements in this page?
   if( m_pagePosition > 0 )
   {
      m_pagePosition--;
      return true;
   }

   // else, we must get the previous element in the parent page.
   // we need to scan a parent page until we have a position which is greater than 0
   MAP_PAGE *page = m_page->m_parent;
   uint16 ppos = m_page->m_parentElement;

   while( page != 0 && ppos == 0 )
   {
      ppos = page->m_parentElement;
      page = page->m_parent;
   }

   // if the page is zero, we can't do anything more
   if( page == 0 )
   {
      // invalidate the iterator
      m_pagePosition = m_map->m_treeOrder;
      return false;
   }

   // if the PPOS is >= count, it means we was in an "higher page"
   if( ppos >= page->m_count )
      ppos = page->m_count - 1;
   else
      ppos--;

   m_page = page;
   m_pagePosition = ppos;

   return true;
}

bool MapIterator::hasNext() const
{
   return m_page != 0 &&
            ( m_page->m_count > m_pagePosition + 1 ||
               m_map->ptrInPage( m_page, m_page->m_count - 1 ) != 0 );
}

bool MapIterator::hasPrev() const
{
   if ( m_page == 0 )
      return false;

   if( m_pagePosition > 0 )
      return true;

   uint16 ppos = m_page->m_parentElement;
   MAP_PAGE *page = m_page->m_parent;


   while( page != 0 && ppos == 0 )
   {
      ppos = page->m_parentElement;
      page = page->m_parent;
   }

   return page != 0;
}

bool MapIterator::equal( const MapIterator &other ) const
{
   return m_map == other.m_map &&
          m_page == other.m_page &&
          m_pagePosition == other.m_pagePosition;
}

//=======================================================
// Map traits

uint32 MapPtrTraits::memSize() const
{
	return sizeof( Map * );
}

void  MapPtrTraits::init( void *itemZone ) const
{
	Map **map = (Map **) itemZone;
	*map = 0;
}

void MapPtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   Map **tgt = (Map **) targetZone;
   Map *src = (Map *) sourceZone;
	*tgt = src;
}

int MapPtrTraits::compare( const void *first, const void *second ) const
{
	return -1;
}

void MapPtrTraits::destroy( void *item ) const
{
// do nothing
}

bool MapPtrTraits::owning() const
{
	return false;
}

void MapPtrOwnTraits::destroy( void *item ) const
{
   Map **ptr = (Map**) item;
   delete (*ptr);
}

bool MapPtrOwnTraits::owning() const
{
	return true;
}

namespace traits
{
	FALCON_DYN_SYM MapPtrTraits &t_MapPtr();
	FALCON_DYN_SYM MapPtrOwnTraits &t_MapPtrOwn();
}

}

/* end of genericmap.cpp */

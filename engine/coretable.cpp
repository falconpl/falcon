/*
   FALCON - The Falcon Programming Language.
   FILE: coretable.cpp

   Table support Iterface for Falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Sep 2008 15:15:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coretable.h>
#include <falcon/traits.h>
#include <falcon/itemtraits.h>
#include <falcon/vm.h>

namespace Falcon {

CoreTableIterator::CoreTableIterator( CoreTable *owner, uint32 pageNum, uint32 itemNum ):
   m_owner( owner ),
   m_pageNum( pageNum ),
   m_itemNum( itemNum )
{}

CoreTableIterator::~CoreTableIterator()
{}

bool CoreTableIterator::next()
{
   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   if ( m_itemNum >= page->length() )
   {
      return false;
   }

   m_itemNum++;
   return true;
}

bool CoreTableIterator::prev()
{
   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   if ( m_itemNum > page->length() )
   {
      return false;
   }

   m_itemNum--;
   return true;
}


bool CoreTableIterator::hasNext() const
{
   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   return m_itemNum < page->length();
}


bool CoreTableIterator::hasPrev() const
{
   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   return m_itemNum <= page->length();
}


Item &CoreTableIterator::getCurrent() const
{
   // oh well, if invalid we should crash, so it hasn't much meaning to
   // check for page...
   CoreArray *page = m_owner->page( m_pageNum );
   fassert( page != 0 );
   fassert( m_itemNum < page->length() );
   return (*page)[m_itemNum];
}


bool CoreTableIterator::isValid() const
{
   if ( m_itemNum < 0 )
      return false;

   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   return m_itemNum < page->length();
}


bool CoreTableIterator::isOwner( void *collection ) const
{
   return ((void *) m_owner ) == collection;
}


bool CoreTableIterator::equal( const CoreIterator &other ) const
{
   if ( ! other.isOwner( m_owner ) )
      return false;

   // same owner, same class...
   CoreTableIterator *tother = (CoreTableIterator *) &other;
   return tother->m_pageNum == m_pageNum && tother->m_itemNum == m_itemNum;
}


bool CoreTableIterator::erase()
{

   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   if ( m_itemNum < page->length() )
   {
      page->remove( m_itemNum );
      return true;
   }
   return false;
}


bool CoreTableIterator::insert( const Item &other )
{
   CoreArray *page = m_owner->page( m_pageNum );
   if( page == 0 )
      return false;

   if ( m_itemNum < page->length() )
   {
      page->insert( other, m_itemNum );
      m_itemNum++;
      return true;
   }
   else if ( m_itemNum == page->length() )
   {
      page->append( other );
      m_itemNum++;
      return true;
   }

   return false;
}


void CoreTableIterator::invalidate()
{
   m_itemNum = noitem;
}


FalconData *CoreTableIterator::clone() const
{
   return new CoreTableIterator( m_owner, m_pageNum, m_itemNum );
}


//===============================================================


CoreTable::CoreTable():
   m_pages(&traits::t_voidp),
   m_pageNumId(noitem),
   m_currentPage(0),
   m_order(noitem),
   m_heading( &traits::t_string, &traits::t_int ),
   m_headerData( &traits::t_item )
{
}

CoreTable::CoreTable( const CoreTable& other ):
   m_pages(other.m_pages),
   m_pageNumId( other.m_pageNumId ),
   m_headerData( other.m_headerData ),
   m_currentPage( other.m_currentPage ),
   m_order( other.m_order ),
   m_heading( other.m_heading )
{
}


CoreTable::~CoreTable()
{}

bool CoreTable::setHeader( CoreArray *header )
{
   if ( m_order != noitem )
   {
      if( m_order != header->length() )
         return false;
   }

   uint32 len = header->length();
   m_headerData.reserve( len );
   m_headerData.resize(0);
   m_heading.clear();

   for( int i = 0; i < len; i++ )
   {
      const Item &itm = (*header)[i];

      // we accept only strings and future bindings
      if ( itm.isFutureBind() )
      {
         // string + value
         m_heading.insert( itm.asLBind(), &i );
         m_headerData.push( &itm.asFBind()->origin() );
      }
      else if ( itm.isString() ) {
         Item nil;
         m_heading.insert( itm.asString(), &i );
         m_headerData.push( &nil );
      }
      else
         return false;
   }

   m_order = len;
   return true;
}


bool CoreTable::insertRow(  CoreArray *ca, uint32 pos, uint32 page )
{
   if ( m_order == 0 || m_order != ca->length() )
      return false;

   CoreArray *tgt;
   if ( page == noitem )
   {
      tgt = currentPage();
      if ( tgt == 0 )
         return false;
   }
   else {
      if( m_pages.size() <= page )
         return false;

      tgt = (CoreArray *) m_pages.at(page);
   }

   if ( pos < ca->length() )
      tgt->insert( ca, pos );
   else
      tgt->append( ca );

   return true;
}


bool CoreTable::removeRow( uint32 pos, uint32 page )
{
   CoreArray *tgt;
   if ( page == noitem )
   {
      tgt = currentPage();
      if ( tgt == 0 )
         return false;
   }
   else {
      if( m_pages.size() <= page )
         return false;

      tgt = (CoreArray *) m_pages.at(page);
   }

   if ( pos >= tgt->length() )
      return false;
   tgt->remove( pos );
   return true;
}


const String &CoreTable::heading( uint32 pos ) const
{
   fassert( pos < order() );

   MapIterator mi = m_heading.begin();
   while( mi.hasCurrent() )
   {
      uint32* p = (uint32*) mi.currentValue();
      if ( *p == pos )
         return *(String *) mi.currentKey();
      mi.next();
   }

   fassert( false );

   // have a nice crash...
   return *(String *) 0;
}


uint32 CoreTable::getHeaderPos( const String &name ) const
{
   uint32* pos = (uint32*) m_heading.find( &name );
   if ( pos == 0 )
      return noitem;
   return *pos;
}


Item *CoreTable::getHeaderData( uint32 pos ) const
{
   if ( pos >= m_headerData.size() )
      return 0;
   return (Item *) m_headerData.at(pos);
}

bool CoreTable::insertPage( CoreArray *data, uint32 pos )
{
   // may be long zero; it's ok
   for( int i = 0; i < data->length(); i ++ )
   {
      if ( ! (*data)[i].isArray() || (*data)[i].asArray()->length() != m_order )
         return false;
   }

   // ok we have a good page to add.
   if( pos < m_pages.size() )
      m_pages.insert( data, pos );
   else
      m_pages.push( data );

   return true;
}

bool CoreTable::removePage( uint32 pos )
{
   return m_pages.remove(pos);
}


const Item &CoreTable::front() const
{
   static Item fake;
   fassert( m_currentPage != 0 );
   fassert( m_currentPage->length() != 0 );

   fake = (*m_currentPage)[0];
   return fake;
}


const Item &CoreTable::back() const
{
   static Item fake;
   fassert( m_currentPage != 0 );
   fassert( m_currentPage->length() != 0 );

   fake = (*m_currentPage)[m_currentPage->length()-1];
   return fake;
}



FalconData *CoreTable::clone() const
{
   return new CoreTable( *this );
}


CoreIterator *CoreTable::getIterator( bool tail )
{
   return new CoreTableIterator( this, m_pageNumId,
      tail ? (
         m_currentPage != 0 && m_currentPage->length() > 0 ?
            currentPage()->length()-1 : 0 ) : 0
         );
}

void CoreTable::clear()
{
   if ( m_currentPage != 0 )
      m_currentPage->resize(0);
}


bool CoreTable::erase( CoreIterator *iter )
{
   return iter->erase();
}


bool CoreTable::insert( CoreIterator *iter, const Item &item )
{
   return iter->insert(item);
}


void CoreTable::gcMark( VMachine *vm )
{
   int i;

   // mark the header data...
   for ( i = 0; i < m_headerData.size(); i ++ )
   {
      vm->memPool()->markItemFast( *((Item *) m_headerData.at(i)) );
   }

   // and all the tables.
   for( i = 0; i < m_pages.size(); i++ )
   {
      Item temp = (CoreArray *) m_pages.at(i);
      vm->memPool()->markItemFast( temp );
   }
}

}

/* end of coretable.cpp */

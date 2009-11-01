/*
   FALCON - The Falcon Programming Language.
   FILE: coretable.cpp

   Table support Interface for Falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Sep 2008 15:15:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/coretable.h>
#include <falcon/iterator.h>
#include <falcon/traits.h>
#include <falcon/itemtraits.h>
#include <falcon/vm.h>

namespace Falcon {

//===============================================================


CoreTable::CoreTable():
   m_currentPage(0),
   m_pages(&traits::t_voidp()),
   m_headerData( &traits::t_item() ),
   m_heading( &traits::t_string(), &traits::t_int() ),
   m_currentPageId(noitem),
   m_order(noitem),
   m_biddingVals(0),
   m_biddingSize(0)
{
}

CoreTable::CoreTable( const CoreTable& other ):
   m_currentPage( other.m_currentPage ),
   m_pages(other.m_pages),
   m_headerData( other.m_headerData ),
   m_heading( other.m_heading ),
   m_currentPageId( other.m_currentPageId ),
   m_order( other.m_order ),
   m_biddingVals(0),
   m_biddingSize(0)
{
}

CoreTable::~CoreTable()
{
   if ( m_biddingVals != 0 ) {
      memFree( m_biddingVals );
      m_biddingVals = 0;
   }
}

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

   for( uint32 i = 0; i < len; i++ )
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

      tgt = *(CoreArray **) m_pages.at(page);
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

      tgt = *(CoreArray **) m_pages.at(page);
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

bool CoreTable::insertPage( CoreObject *self, CoreArray *data, uint32 pos )
{
   uint32 i;

   // may be long zero; it's ok
   for( i = 0; i < data->length(); i ++ )
   {
      if ( ! (*data)[i].isArray() || (*data)[i].asArray()->length() != m_order )
         return false;
   }

   for( i = 0; i < data->length(); i ++ )
   {
      (*data)[i].asArray()->table( self );
   }

   // ok we have a good page to add.
   if( pos < m_pages.size() )
      m_pages.insert( data, pos );
   else
      m_pages.push( data );

   if ( m_currentPageId >= pos )
      m_currentPageId++;

   return true;
}

bool CoreTable::removePage( uint32 pos )
{
   if ( m_pages.size() == 1 || pos >= m_pages.size() )
   {
      // can't delete the only page left.
      return false;
   }

   // declare the page dead
   page(pos)->gcMark(1);

   // are we going to remove the current page?
   if ( m_currentPageId == pos )
   {
      m_pages.remove(pos);
      m_currentPage = *(CoreArray **) m_pages.at(0);
      m_currentPageId = 0;
   }
   else {
      m_pages.remove(pos);
   }

   // If it was equal, it became 0
   if( m_currentPageId > pos )
      m_currentPageId--;

   return true;
}


const Item &CoreTable::front() const
{
   static Item fake;
   fassert( m_currentPage != 0 );
   fassert( m_currentPage->length() != 0 );

   fake = (*m_currentPage)[0];
   return fake;
}


void CoreTable::append( const Item &data )
{
   if ( ! data.isArray() )
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "A" ) );

   insertRow( data.asArray() );
}

/** Prepend an item at the beginning of the sequence. */
void CoreTable::prepend( const Item &data )
{
      if ( ! data.isArray() )
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "A" ) );

   insertRow( data.asArray(), 0 );
}

const Item &CoreTable::back() const
{
   static Item fake;
   fassert( m_currentPage != 0 );
   fassert( m_currentPage->length() != 0 );

   fake = (*m_currentPage)[m_currentPage->length()-1];
   return fake;
}



CoreTable *CoreTable::clone() const
{
   return new CoreTable( *this );
}


void CoreTable::clear()
{
   if ( m_currentPage != 0 )
      m_currentPage->resize(0);
}


void CoreTable::gcMark( uint32 gen )
{
   uint32 i;

   // mark the header data...
   for ( i = 0; i < m_headerData.size(); i ++ )
   {
      memPool->markItem( *((Item *) m_headerData.at(i)) );
   }

   // and all the tables.
   for( i = 0; i < m_pages.size(); i++ )
   {
      CoreArray* page = *(CoreArray**)m_pages.at(i);
      page->gcMark( gen );
   }
}

void CoreTable::reserveBiddings( uint32 size )
{
   if ( size > m_biddingSize )
   {
      if ( m_biddingVals != 0 )
         memFree( m_biddingVals );
      m_biddingSize = size;
      m_biddingVals = (numeric *) memAlloc( size * sizeof( numeric ) );
   }
}

void CoreTable::renameColumn( uint32 pos, const String &name )
{
   fassert( pos < order() );

   MapIterator mi = m_heading.begin();
   while( mi.hasCurrent() )
   {
      uint32* p = (uint32*) mi.currentValue();
      if ( *p == pos ) {
         m_heading.erase( mi );
         m_heading.insert( &name, &pos );
         return;
      }
      mi.next();
   }

   fassert( false );

}

void CoreTable::insertColumn( uint32 pos, const String &name, const Item &data, const Item &dflt )
{
   // if pos >= order, we're lucky. Append.
   if ( pos >= m_order )
   {
      pos = m_order;
      m_heading.insert( &name, &pos );
   }
   else
   {
      // first, move all following elements 1 forward
      MapIterator mi = m_heading.begin();
      while( mi.hasCurrent() )
      {
         uint32* p = (uint32*) mi.currentValue();
         if ( *p >= pos ) {
            *p = *p+1;
         }
         mi.next();
      }
      // then insert the new entry
      m_heading.insert( &name, &pos );
   }

   m_order++;
   // add the column data.
   m_headerData.insert( (void *) &data, pos );

   // now, for each page, for each row, insert the default item.
   for( uint32 pid = 0; pid < m_pages.size(); pid++ )
   {
      CoreArray *pg = page( pid );
      for( uint32 rowid = 0; rowid < pg->length(); rowid++ )
      {
         CoreArray *row = pg->at(rowid).asArray();
         // modify is forbidden if the array has a table.
         CoreObject *save = row->table();
         row->table(0);
         row->insert( dflt, pos );
         row->table(save);
      }
   }
}

bool CoreTable::removeColumn( uint32 pos )
{
   // if pos >= order, we're lucky. Append.
   if ( pos >= m_order )
   {
     return false;
   }

   // first, move all following elements 1 forward
   MapIterator mi = m_heading.begin();
   while( mi.hasCurrent() )
   {
      uint32* p = (uint32*) mi.currentValue();

      // when we find the foobar'd column, remove it.
      if ( *p == pos )
      {
         m_heading.erase( mi );
         continue;
      }

      // else, take other columns back
      if ( *p > pos ) {
         *p = *p-1;
      }
      mi.next();
   }

   // add the column data.
   m_headerData.remove( pos );
   m_order--;

   // now, for each page, for each row, insert the default item.
   for( uint32 pid = 0; pid < m_pages.size(); pid++ )
   {
      CoreArray *pg = page( pid );
      for( uint32 rowid = 0; rowid < pg->length(); rowid++ )
      {
         CoreArray *row = pg->at(rowid).asArray();
         // modify is forbidden if the array has a table.
         CoreObject *save = row->table();
         row->table(0);
         row->remove( pos );
         row->table(save);
      }
   }

   return true;
}


//============================================
// Iterator implementation
//============================================

void CoreTable::getIterator( Iterator& tgt, bool tail ) const
{
   // give up the ownership of the iterator to the current page.
   tgt.sequence( &m_currentPage->items() );
   m_currentPage->items().getIterator( tgt, tail );
}

void CoreTable::copyIterator( Iterator& tgt, const Iterator& source ) const
{
   // actually never called
}

void CoreTable::insert( Iterator &iter, const Item &data )
{
   // actually never called
}

void CoreTable::erase( Iterator &iter )
{
   // actually never called
}

bool CoreTable::hasNext( const Iterator &iter ) const
{
   // actually never called
   return false;
}

bool CoreTable::hasPrev( const Iterator &iter ) const
{
   // actually never called
   return false;
}

bool CoreTable::hasCurrent( const Iterator &iter ) const
{
   // actually never called
   return false;
}

bool CoreTable::next( Iterator &iter ) const
{
   // actually never called
   return false;
}

bool CoreTable::prev( Iterator &iter ) const
{
   // actually never called
   return false;
}

Item& CoreTable::getCurrent( const Iterator &iter )
{
   // actually never called
   throw new CodeError( ErrorParam( e_invalid_iter, __LINE__ ) );
}

Item& CoreTable::getCurrentKey( const Iterator &iter )
{
   // actually never called
   throw new CodeError( ErrorParam( e_invalid_iter, __LINE__ ) );
}

bool CoreTable::equalIterator( const Iterator &first, const Iterator &second ) const
{
   // actually never called
   return false;
}


}

/* end of coretable.cpp */

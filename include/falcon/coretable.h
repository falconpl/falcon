/*
   FALCON - The Falcon Programming Language.
   FILE: coretable.h

   Table support Iterface for Falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Sep 2008 15:15:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_CORE_TABLE_H
#define FLC_CORE_TABLE_H

#include <falcon/setup.h>
#include <falcon/carray.h>
#include <falcon/genericvector.h>
#include <falcon/sequence.h>
#include <falcon/fassert.h>
#include <falcon/item.h>

namespace Falcon {

class CoreTable;

class CoreTableIterator: public CoreIterator
{
   CoreTable *m_owner;
   uint32 m_pageNum;
   uint32 m_itemNum;

   enum {
      noitem = 0xFFFFFFFF
   };

   friend class CoreTable;
public:

   CoreTableIterator( CoreTable *owner, uint32 pageNum, uint32 itemNum );
   virtual ~CoreTableIterator();

   virtual bool next();
   virtual bool prev();
   virtual bool hasNext() const;
   virtual bool hasPrev() const;

   virtual Item &getCurrent() const;

   virtual bool isValid() const;
   virtual bool isOwner( void *collection ) const;
   virtual bool equal( const CoreIterator &other ) const;
   virtual bool erase();
   virtual bool insert( const Item &other );

   virtual void invalidate();

   virtual FalconData *clone() const;

   // specific interface
   CoreTable *table() const { return m_owner; }
   uint32 pageNum() const { return m_pageNum; }
   uint32 itemNum() const { return m_itemNum; }
};


//=========================================================
//

class CoreTable: public Sequence
{
   CoreArray *m_currentPage;
   GenericVector m_pages;
   GenericVector m_headerData;
   Map m_heading;
   uint32 m_pageNumId;
   uint32 m_order;

   /** Used for votation and bidding operations */
   numeric *m_biddingVals;
   uint32 m_biddingSize;

public:
   enum {
      noitem = 0xFFFFFFFF
   };

   CoreTable();
   CoreTable( const CoreTable& other );
   virtual ~CoreTable();

   CoreArray *page( uint32 num ) const {
      return *reinterpret_cast<CoreArray **>(m_pages.size() >= num ? m_pages.at(num) : 0);
   }

   uint32 pageCount() const { return m_pages.size(); }

   /** Returns the order (number of colums) in the table */
   uint32 order() const { return m_order; }

   /** Returns nth column-wide data in the table (const).
      Using an incorrect (greater than order) position parameter will crash.
   */
   const Item& columnData( uint32 pos ) const {
      fassert( pos < order() );
      return *(Item*) m_headerData.at(pos);
   }

   /** Returns nth column-wide data in the table (const).
      Using an incorrect (greater than order) position parameter will crash.
   */
   Item& columnData( uint32 pos ) {
      fassert( pos < order() );
      return *(Item*) m_headerData.at(pos);
   }

   /** Returns the nth heading (column title).
      Using an incorrect (greater than order) position parameter will crash.
   */
   const String &heading( uint32 pos ) const;

   /** Rename a colunm.
      \param pos Column to be renamed.
      \param name The new name for the given column.
   */
   void renameColumn( uint32 pos, const String &name );

   /** Inserts a colum in the table.
      This alters the heading and all the pages, inserting a row of default
      items (can be nil) as the new column.

      If pos >= order, the column will be appended at end.
      \param pos The position where to add the column.
      \param name The name of the column to be inserted.
      \param data The column data, can be nil.
      \param dflt The default value for inserted items; can be nil.
   */
   void insertColumn( uint32 pos, const String &name, const Item &data, const Item &dflt );

   /** Removes a colum from the table.
      This alters the heading and all the pages, removing the item
      at given position in all the arrays in every page.

      \param pos The index of the column to be removed.
      \return true on success, false if pos >= order.
   */
   bool removeColumn( uint32 pos );

   bool setCurrentPage( uint32 num )
   {
      if ( num < m_pages.size() )
      {
         m_pageNumId = num;
         m_currentPage = *reinterpret_cast<CoreArray **>(m_pages.at( num ));
         return true;
      }

      return false;
   }

   CoreArray *currentPage() const { return m_currentPage; }
   bool setHeader( CoreArray *header );
   uint32 getHeaderPos( const String &name ) const;
   Item *getHeaderData( uint32 pos ) const;

   Item *getHeaderData( const String &name ) const {
      uint32 pos = getHeaderPos( name );
      if ( pos == noitem )
         return 0;

      return getHeaderData( pos );
   }

   /** Inserts a new row in a page.
      If the core array has not the same length of the order of this table, the
      function will fail.

      \param pos The position at which to insert the row; will add it if position >= length.
      \param ca The array to be added.
      \param page The page into which the row must be added, set
      \return true on success
   */
   bool insertRow( CoreArray *ca, uint32 pos=noitem, uint32 page = noitem );

   /** Removes a row.
      \param pos The position at which to remove the row.
      \param page The page into which the row must be remove, set
      \return true on success
   */
   bool removeRow( uint32 pos, uint32 page = noitem );

   /** Inserts a page.
      The function checks if the array is actually a matrix with an order compatible
      with the table.
      \param pos Where to insert the page. If greater than the number of pages, or
         set to noitem, will append the page at the end.

      \param array An array to fill the page. Can be empty; otherwise, every element must
         be an array of the same order of this table.

      \return true if the page can be added, false if the given array wasn't of correct order.
   */
   bool insertPage( CoreArray *data, uint32 pos = noitem );

   /** Removes a page.
      \param pos Which page to remove.
      \return true if the page can be remove, false the position is invalid.
   */
   bool removePage( uint32 pos );

   //========================================
   //===== Sequence interface ===============
   //
   /** Deletes the list.
      Items are shallowly destroyed.
   */
   virtual FalconData *clone() const;

   virtual const Item &front() const;
   virtual const Item &back() const;

   /** Creates an iterator item for the object.
      The CoreTableIterator is an instance of the CoreIterator class and can be used
      as a part of the VM iterator system.
      This method returns a newly created CoreTableIterator pointing to the first or
      last row of the current page in this table.
      If the table is empty, the item will be declared invalid.

      \param tail true to get the iterator to the last element in the list.
      \return a newly created iterator.
   */
   virtual CoreIterator *getIterator( bool tail=false );


   /** Removes all the raws in the current page. */
   virtual void clear();

   /** Implementing sequence interface */
   virtual bool erase( CoreIterator *iter );

   /** Implementing sequence interface */
   virtual bool insert( CoreIterator *iter, const Item &item );

   /** Tells if the list is empty.
      \return true if the list is empty.
   */
   virtual bool empty() const { return m_currentPage == 0 || m_currentPage->length() == 0; }

   /** Perform marking of items stored in the table.
   */
   virtual void gcMark( VMachine * );

   /** Returns bidding results.
      Must be initialized with makebiddings().
   */
   numeric *biddings() const { return m_biddingVals; }

   /** Returns max size of the biddings array. */
   uint32 biddingsSize() const { return m_biddingSize; }

   void reserveBiddings( uint32 size );
};

}

#endif

/* end of coretable.h */

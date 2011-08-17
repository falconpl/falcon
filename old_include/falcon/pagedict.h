/*
   FALCON - The Falcon Programming Language.
   FILE: pagedict.h

   Item dictionary - paged dictionary version and related utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core dictionary - paged dictionary version and related utilities.
*/

#ifndef flc_pagedict_H
#define flc_pagedict_H

#include <falcon/types.h>
#include <falcon/itemdict.h>
#include <falcon/item.h>
#include <falcon/itemtraits.h>
#include <falcon/genericmap.h>
#include <stdlib.h>

#define  flc_DICT_GROWTH  16

namespace Falcon
{

class PageDict;
class VMachine;
class Iterator;


class FALCON_DYN_CLASS PageDict: public ItemDict
{
   ItemTraits m_itemTraits;
   Map m_map;
   uint32 m_mark;

   static void PageDictIterDeletor( Iterator* iter );
public:

   PageDict();
   PageDict( uint32 pageSize );
   ~PageDict();

   virtual uint32 length() const;
   virtual Item *find( const Item &key ) const;
   virtual bool findIterator( const Item &key, Iterator &iter );
   virtual void smartInsert( const Iterator &iter, const Item &key, const Item &value );

   virtual const Item &front() const;
   virtual const Item &back() const;
   virtual void append( const Item& item );
   virtual void prepend( const Item& item );

   virtual bool remove( const Item &key );
   virtual void put( const Item &key, const Item &value );

   virtual PageDict *clone() const;
   virtual void merge( const ItemDict &dict );
   virtual void clear();
   virtual bool empty() const;
   void gcMark( uint32 gen );

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

/* end of pagedict.h */

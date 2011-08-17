/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cdict.h

   Core dictionary and related utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core dictionary and related utilities.
*/

#ifndef flc_lineardict_H
#define flc_lineardict_H

#include <falcon/types.h>
#include <falcon/itemdict.h>
#include <falcon/item.h>
#include <stdlib.h>

#define  flc_DICT_GROWTH  16

namespace Falcon {

class LinearDict;
class VMachine;
class Iterator;

/** Class representing one element in a dictionary. */
class FALCON_DYN_CLASS LinearDictEntry
{
   Item m_key;
   Item m_value;

public:

   LinearDictEntry( const Item &k, const Item &v ):
      m_key( k ),
      m_value( v )
   {}

   void key( const Item &v ) { m_key = v; }
   const Item &key() const { return m_key; }
   const Item &value() const { return m_value; }
   Item &key() { return m_key; }
   Item &value() { return m_value; }
   void value( const Item &v ) { m_value = v; }

   friend class LinearDict;
};


class FALCON_DYN_CLASS LinearDict: public ItemDict
{
   uint32 m_size;
   uint32 m_alloc;
   uint32 m_invalidPos;
   LinearDictEntry *m_data;
   uint32 m_mark;

   bool addInternal( uint32 pos, const Item &key, const Item &value );

   /** Make a search in a standard dictionary.
      This function returns true if the required item is found,
      and false if the item is not found. In the first case ret_pos becomes
      the position in the dictionary array where the item is stored,
      while in the second case it is set to the best position where a new
      item with the not present key may be inserted.
   */
   bool findInternal( const Item &key, uint32 &ret_pos ) const;
   bool removeAt( uint32 pos );

public:

   LinearDict();
   LinearDict( uint32 prealloc );
   ~LinearDict();
   virtual LinearDict *clone() const;
   virtual void gcMark( uint32 gen );

   virtual uint32 length() const;
   virtual Item *find( const Item &key ) const;
   virtual bool findIterator( const Item &key, Iterator &iter );

   virtual const Item &front() const;
   virtual const Item &back() const;
   virtual void append( const Item& item );
   virtual void prepend( const Item& item );

   virtual bool remove( const Item &key );
   virtual void put( const Item &key, const Item &value );
   virtual void smartInsert( const Iterator &iter, const Item &key, const Item &value );

   virtual void merge( const ItemDict &dict );
   virtual void clear();
   virtual bool empty() const;

   uint32 esize( uint32 num ) const { return sizeof( LinearDictEntry ) * num; }

   LinearDictEntry *entries() const { return m_data; }
   uint32 allocated() const { return m_alloc; }

   void length( uint32 size ) { m_size = size; }
   void allocated( uint32 size ) { m_alloc = size; }
   void entries( LinearDictEntry *d ) { m_data = d; }

   LinearDictEntry *elementAt( uint32 pos ) const
   {
      if ( pos >= length() )
         return 0;
      return entries() + pos;
   }

   bool find( const Item &key, uint32 &ret_pos ) const {
      return findInternal( key, ret_pos );
   }

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
   virtual bool onCriterion( Iterator* elem ) const;
};

}

#endif

/* end of flc_cdict.h */

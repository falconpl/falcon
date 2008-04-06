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
#include <falcon/cdict.h>
#include <falcon/item.h>
#include <stdlib.h>

#define  flc_DICT_GROWTH  16

namespace Falcon {

class LinearDict;
class VMachine;

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


class LinearDictIterator: public DictIterator
{
   uint32 m_dictPos;
   LinearDict *m_dict;
   uint32 m_versionNumber;

public:
   LinearDictIterator( LinearDict *owner, uint32 pos );

   virtual bool next();
   virtual bool prev();
   virtual bool hasNext() const;
   virtual bool hasPrev() const;
   virtual Item &getCurrent() const;
   virtual const Item &getCurrentKey() const;

   virtual bool isValid() const;
   virtual bool isOwner( void *collection ) const;
   virtual void invalidate();
   virtual bool equal( const CoreIterator &other ) const;
   virtual bool erase();
   virtual bool insert( const Item &data );

   friend class LinearDict;
};

class FALCON_DYN_CLASS LinearDict: public CoreDict
{
   uint32 m_size;
   uint32 m_alloc;
   LinearDictEntry *m_data;
   uint32 m_version;
   uint32 m_travPos;

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

   LinearDict( VMachine *vm );
   LinearDict( VMachine *vm, uint32 prealloc );
   ~LinearDict();

   virtual uint32 length() const;
   virtual Item *find( const Item &key );
   virtual bool find( const Item &key, DictIterator &iter );
   virtual DictIterator *findIterator( const Item &key );

   virtual bool remove( DictIterator &iter );
   virtual bool remove( const Item &key );
   virtual void insert( const Item &key, const Item &value );
   virtual void smartInsert( DictIterator &iter, const Item &key, const Item &value );

   virtual void first( DictIterator &iter );
   virtual void last( DictIterator &iter );
   virtual DictIterator *first();
   virtual DictIterator *last();

   virtual bool equal( const CoreDict &other ) const;
   virtual CoreDict *clone() const;
   virtual void merge( const CoreDict &dict );
   virtual void clear();

   virtual void traverseBegin();
   virtual bool traverseNext( Item &key, Item &value );


   uint32 version() const { return m_version; }
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
};

}

#endif

/* end of flc_cdict.h */

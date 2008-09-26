/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cdict.h

   Core dictionary - paged dictionary version and related utilities.
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
#include <falcon/cdict.h>
#include <falcon/item.h>
#include <falcon/itemtraits.h>
#include <falcon/genericmap.h>
#include <stdlib.h>

#define  flc_DICT_GROWTH  16

namespace Falcon {

class PageDict;
class VMachine;

class PageDictIterator: public DictIterator
{
   MapIterator m_iter;
   uint32 m_versionNumber;
   PageDict *m_owner;

   PageDictIterator( PageDict *owner, const MapIterator &iter );
public:


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
   virtual bool insert( const Item &item );

   virtual FalconData *clone() const;

   friend class PageDict;
};

class FALCON_DYN_CLASS PageDict: public CoreDict
{
   VMItemTraits m_itemTraits;
   MapIterator m_traverseIter;
   Map m_map;
   uint32 m_version;

public:

   PageDict( VMachine *vm );
   PageDict( VMachine *vm, uint32 pageSize );
   ~PageDict();

   virtual uint32 length() const;
   virtual Item *find( const Item &key ) const;
   virtual bool find( const Item &key, DictIterator &iter );
   virtual DictIterator *findIterator( const Item &key );
   virtual void smartInsert( DictIterator &iter, const Item &key, const Item &value );

   virtual bool remove( DictIterator &iter );
   virtual bool remove( const Item &key );
   virtual void insert( const Item &key, const Item &value );

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
};

}

#endif

/* end of flc_cdict.h */

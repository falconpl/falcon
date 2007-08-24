/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cdict.h
   $Id: pagedict.h,v 1.2 2007/08/11 00:11:52 jonnymind Exp $

   Core dictionary - paged dictionary version and related utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

   friend class PageDict;
};

class FALCON_DYN_CLASS PageDict: public CoreDict
{
   VMItemTraits m_itemTraits;
   MapIterator m_traverseIter;
   uint32 m_version;
   Map m_map;

public:

   PageDict( VMachine *vm );
   PageDict( VMachine *vm, uint32 pageSize );
   ~PageDict();

   virtual uint32 length() const;
   virtual Item *find( const Item &key );
   virtual DictIterator *findIterator( const Item &key );
   virtual bool remove( DictIterator *iter );
   virtual bool remove( const Item &key );
   virtual void insert( const Item &key, const Item &value );
   virtual DictIterator *begin();
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

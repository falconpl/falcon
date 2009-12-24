/*
   FALCON - The Falcon Programming Language.
   FILE: coredict.h

   Core dictionary -- base abstract class for dictionary interfaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Aug 2009 21:18:28 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core dictionary -- base abstract class for dictionary interfaces.
*/

#ifndef FALCON_ITEM_DICT_H
#define FALCON_ITEM_DICT_H

#include <falcon/types.h>
#include <falcon/sequence.h>

namespace Falcon {

class Item;

/** Base class for item dictionaries.
 * This is the base class for item dictionaries. Dictionaries
 * must support the sequence interface. They cannot be immediately
 * stored into falcon Items; a CoreDict wrapper is necessary.
 */
class FALCON_DYN_CLASS ItemDict: public Sequence
{
public:
   /** Override sequence to inform all that we're a dictionary. */
   virtual bool isDictionary() const { return true; }

   virtual uint32 length() const = 0;
   virtual Item *find( const Item &key ) const = 0;
   virtual bool findIterator( const Item &key, Iterator &iter ) = 0;

   virtual bool remove( const Item &key ) = 0;
   virtual void put( const Item &key, const Item &value ) = 0;
   virtual void smartInsert( const Iterator &iter, const Item &key, const Item &value ) = 0;

   virtual void merge( const ItemDict &dict ) = 0;
   virtual void clear() = 0;

   virtual int compare( const ItemDict& other ) const { return compare( other, 0); }
   
private:

   /** Classed used internally to track loops in traversals. */
   class Parentship
   {
   public:
      const ItemDict* m_dict;
      Parentship* m_parent;
      
      Parentship( const ItemDict* d, Parentship* parent=0 ):
         m_dict(d),
         m_parent( parent )
      {}
   };

   int compare( const ItemDict& other, Parentship* p ) const;
   int checkValue( const Item& first, const Item& second, Parentship& current ) const;
};

}

#endif /* FALCON_ITEM_DICT_H */

/*
   FALCON - The Falcon Programming Language.
   FILE: coredict.h

   Core dictionary -- base abstract class for dictionary interfaces.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core dictionary -- base abstract class for dictionary interfaces.
*/

#ifndef FALCON_CORE_DICT_H
#define FALCON_CORE_DICT_H

#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/itemdict.h>
#include <falcon/deepitem.h>

namespace Falcon {

class Item;

class FALCON_DYN_CLASS CoreDict: public DeepItem, public Garbageable
{
   bool m_blessed;
   ItemDict* m_dict;

public:
   CoreDict( ItemDict* dict ):
      m_blessed( false ),
      m_dict( dict )
   {
      m_dict->owner( this );
   }

   CoreDict( const CoreDict& other ):
      m_blessed( other.m_blessed ),
      m_dict( (ItemDict*) other.m_dict->clone() )
   {
      m_dict->owner( this );
   }

   virtual ~CoreDict()
   {
      delete m_dict;
   }

   const ItemDict& items() const { return *m_dict; }
   ItemDict& items() { return *m_dict; }

   uint32 length() const { return m_dict->length(); }

   Item *find( const Item &key ) const { return m_dict->find( key ); }
   bool findIterator( const Item &key, Iterator &iter ) { return m_dict->findIterator( key, iter ); }

   bool remove( const Item &key ) { return m_dict->remove( key ); }
   void put( const Item &key, const Item &value ) { return m_dict->put( key, value ); }
   void smartInsert( const Iterator &iter, const Item &key, const Item &value ) {
      return m_dict->smartInsert( iter, key, value );
   }

   CoreDict *clone() const { return new CoreDict( *this ); }
   void merge( const CoreDict &dict ) { m_dict->merge( *dict.m_dict ); }
   void clear() { m_dict->clear(); }
   
   int compare( const CoreDict* other ) const { return items().compare( other->items() ); } 

   /** Performs a find using a static string as a key.
       This wraps the string in a temporary item and calls
       the normal find(const Item &)
    */
   Item *find( const String &key ) const;

   //=======================================
   // Utilities

   bool find( const Item &key, Item &value );

   bool empty() const { return length() == 0; }

   /** Returns true if this dictionary is blessed. */
   bool isBlessed() const { return m_blessed; }
   
   /** Returns a method out of this dictionary.
   
      This method returns true if there is a function item
      stored under the given key, provided that this
      dictionary is blessed.
      
      \param name The name of the method to be searched.
      \param mth An item that will receive a full readied method on success.
      \return true if the method was found.
   */
   bool getMethod( const String& name, Item &mth );

   /** Bless this dictionary.
      A blessed dictionary becomes a flessible instance.

      Its elements are treated as properties; the dot accessor will
      act as searching for the given property, and a read dot
      accessor will create a mehtod; in the method, the dictionary
      can be accessed through "self".
   */
   void bless( bool b ) { m_blessed = b; }

   virtual void readProperty( const String &, Item &item );
   virtual void writeProperty( const String &, const Item &item );
   virtual void readIndex( const Item &pos, Item &target );
   virtual void writeIndex( const Item &pos, const Item &target );

   virtual void gcMark( uint32 gen );
};

}

#endif

/* end of coredict.h */

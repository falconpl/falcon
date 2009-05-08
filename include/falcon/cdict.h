/*
   FALCON - The Falcon Programming Language.
   FILE: flc_cdict.h

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

#ifndef flc_cdict_H
#define flc_cdict_H

#include <falcon/types.h>
#include <falcon/citerator.h>
#include <falcon/garbageable.h>
#include <falcon/item.h>

namespace Falcon {

class FALCON_DYN_CLASS DictIterator: public CoreIterator
{
protected:
   DictIterator() {}

public:
   virtual const Item &getCurrentKey() const = 0;
};



class FALCON_DYN_CLASS CoreDict: public DeepItem, public Garbageable
{
   bool m_blessed;

protected:
   CoreDict():
      Garbageable(),
      m_blessed( false )
   {}

   CoreDict( uint32 alloc ):
      Garbageable(),
      m_blessed( false )
   {}

public:
   //=======================================
   // Public overridable interface
   //

   virtual uint32 length() const =0;
   /** Performs a find using a static string as a key.
      This wraps the string in a temporary item and calls
      the normal find(const Item &)
   */
   Item *find( const String &key ) const;
   virtual Item *find( const Item &key ) const = 0;
   virtual bool find( const Item &key, DictIterator &iter ) = 0;
   virtual DictIterator *findIterator( const Item &key ) = 0;

   virtual bool remove( DictIterator &iter ) = 0;
   virtual bool remove( const Item &key ) = 0;
   virtual void insert( const Item &key, const Item &value ) = 0;
   virtual void smartInsert( DictIterator &iter, const Item &key, const Item &value ) = 0;

   virtual void first( DictIterator &iter ) = 0;
   virtual void last( DictIterator &iter ) = 0;
   virtual DictIterator *first() = 0;
   virtual DictIterator *last() = 0;

   virtual bool equal( const CoreDict &other ) const = 0;
   virtual CoreDict *clone() const = 0;
   virtual void merge( const CoreDict &dict ) = 0;
   virtual void clear() = 0;


   /** Generic traversal interface.
      Usually, dictionary traversal is needed by VM or other engine related classes.
   */
   virtual void traverseBegin() = 0;
   virtual bool traverseNext( Item &key, Item &value ) = 0;

   //=======================================
   // Utilities

   bool find( const Item &key, Item &value )
   {
      Item *itm;
      if( ( itm = find( key ) ) != 0 )
      {
         value = *itm;
         return true;
      }
      return false;
   }

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
};

}

#endif

/* end of flc_cdict.h */

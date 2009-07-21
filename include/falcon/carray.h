/*
   FALCON - The Falcon Programming Language.
   FILE: flc_carray.h

   Core array
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab dic 4 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Core array.
*/

#ifndef flc_flc_carray_H
#define flc_flc_carray_H

#include <falcon/types.h>
#include <falcon/garbageable.h>
#include <falcon/item.h>
#include <falcon/deepitem.h>
#include <falcon/sequence.h>

#define flc_ARRAY_GROWTH   128

namespace Falcon {

class Item;
class Bindings;
class CoreArray;

class FALCON_DYN_CLASS ItemArray: public Sequence
{
   uint32 m_alloc;
   uint32 m_size;
   Item *m_data;

   friend class CoreArray;

   ItemArray( Item *buffer, uint32 size, uint32 alloc );

public:
   ItemArray();
   ItemArray( const ItemArray& other );
   ItemArray( uint32 prealloc );

   virtual ~ItemArray();

   virtual const Item &front() const { return m_data[0]; }
   virtual const Item &back() const { return m_data[m_size-1]; }

   Item *elements() const { return m_data; }
   void elements( Item *d ) { m_data = d; }
   uint32 allocated() const { return m_alloc; }
   uint32 length()  const { return m_size; }
   void length( uint32 size ) { m_size = size; }
   void allocated( uint32 size ) { m_alloc = size; }

   virtual CoreIterator *getIterator( bool tail = false ) { return 0; }
   virtual bool insert( CoreIterator *iter, const Item &data ) { return true; }
   virtual bool erase( CoreIterator *iter ) { return true; }
   virtual void clear() { m_size = 0; }
   virtual bool empty() const { return m_size == 0; }

   virtual void gcMark( uint32 mark );
   virtual FalconData *clone() const ;

   virtual void append( const Item &ndata );
   virtual void prepend( const Item &ndata );

   void merge( const ItemArray &other );
   void merge_front( const ItemArray &other );

   bool insert( const Item &ndata, int32 pos );
   bool insert( const ItemArray &other, int32 pos );
   bool remove( int32 pos );
   bool remove( int32 first, int32 last );
   bool change( const ItemArray &other, int32 begin, int32 end );
   int32 find( const Item &itm ) const;
   bool insertSpace( uint32 pos, uint32 size );

   void resize( uint32 size );
   /**
    * Reduce the memory used by this array to exactly its size.
    */
   void compact();
   void reserve( uint32 size );

   ItemArray *partition( int32 start, int32 end ) const;

   inline virtual const Item &at( int32 pos ) const
   {
      if ( pos < 0 )
         pos = m_size + pos;
      if ( pos < 0 || pos > (int32) m_size )
         throw "Invalid range while accessing Falcon::CoreArray";
      return m_data[pos];
   }

   inline virtual Item &at( int32 pos )
   {
      if ( pos < 0 )
         pos = m_size + pos;
      if ( pos < 0 || pos > (int32)m_size )
         throw "Invalid range while accessing Falcon::CoreArray";
      return m_data[pos];
   }

   inline Item &operator[]( int32 pos ) throw()
   {
      return m_data[pos];
   }

   inline const Item &operator[]( int32 pos ) const throw()
   {
      return m_data[pos];
   }


   /** An inline utility to compute element size.
    *
    * @param count numbrer of elements
    * @return the amout of bytes needed to store the elements
    */
   int32 esize( int32 count=1 ) const { return sizeof( Item ) * count; }

};


/** Core array (or array of items).
*/

class FALCON_DYN_CLASS CoreArray:  public DeepItem, public Garbageable
{
   ItemArray m_itemarray;
   CoreDict *m_bindings;
   CoreObject *m_table;
   uint32 m_tablePos;

   CoreArray( Item *buffer, uint32 size, uint32 alloc );

public:

   /** Creates the core array. */
   CoreArray();
   CoreArray( const CoreArray& other );
   CoreArray( uint32 prealloc );

   ~CoreArray();

   const ItemArray& items() const { return m_itemarray; }
   ItemArray& items() { return m_itemarray; }

   void append( const Item &ndata ) {
      if ( m_table != 0 )
         return;
      m_itemarray.append( ndata );
   }
   void prepend( const Item &ndata ) {
      if ( m_table != 0 )
         return;

      m_itemarray.prepend( ndata );
   }

   void merge( const CoreArray &other ) {
      if ( m_table != 0 )
         return;
      m_itemarray.merge( other.m_itemarray );
   }

   void merge_front( const CoreArray &other ) {
      if ( m_table != 0 )
         return;
      m_itemarray.merge( other.m_itemarray );
   }

   bool insert( const Item &ndata, int32 pos ) {
      if ( m_table != 0 )
         return false;
      return m_itemarray.insert( ndata, pos );
   }

   bool insert( const CoreArray &other, int32 pos ) {
      if ( m_table != 0 )
         return false;
      return m_itemarray.insert( other.m_itemarray, pos );
   }

   bool remove( int32 pos ) {
      if ( m_table != 0 )
         return false;
      return m_itemarray.remove( pos );
   }

   bool remove( int32 first, int32 last ) {
      if ( m_table != 0 )
         return false;
      return m_itemarray.remove( first, last );
   }

   bool change( const CoreArray &other, int32 begin, int32 end ) {
      if ( m_table != 0 )
         return false;

      return m_itemarray.change( other.m_itemarray, begin, end );
   }

   int32 find( const Item &itm ) const { return m_itemarray.find( itm ); }

   bool insertSpace( uint32 pos, uint32 size ) {
      if ( m_table != 0 )
         return false;
      return m_itemarray.insertSpace( pos, size );
   }

   void resize( uint32 size ) {
      if ( m_table != 0 )
         return;
      m_itemarray.resize( size );
   }

   void reserve( uint32 size ) {
      m_itemarray.reserve( size );
   }

   CoreArray *partition( int32 start, int32 end ) const;
   CoreArray *clone() const;

   uint32 length() const { return m_itemarray.length(); }
   void length( uint32 size ) { return m_itemarray.length( size ); }

   /** Create the bindings for this array, or get those already created. */
   CoreDict *makeBindings();
   CoreDict *bindings() const { return m_bindings; }
   void setBindings( CoreDict *binds ) { m_bindings = binds; }

   /** Gets a proprty of this vector.
      Properties are either bindings or table properties. Bindings
      override table-wide properties.

      If the given property is not found, 0 is returned.
      \param name The property to be found.
      \return the property item if found or zero.
   */
   Item* getProperty( const String &name );

   /** Set a property in this array.
      If there is a biniding with the given property name, that item is updated.
      If not, If there is a table with a column name, the coresponding item in
      the array is updated.
      If not, new bindings are created, and the property is stored as a new binding.

      \param name The property to be updated.
      \param data The update data.
   */
   void setProperty( const String &name, const Item &data );

   /** Checks the position to be in the array, and eventually changes it if it's negative.
      \param pos the position to be checked and eventually turned into a positive value.
      \return false if pos is outside the array size
   */
    bool checkPosBound( int32 &pos )
    {
      register int s = length();
      if ( pos < 0 )
         pos = s + pos;
      if ( pos < 0 || pos >= s )
         return false;
      return true;
    }

   const Item &at( int32 pos ) const
   {
      return m_itemarray.at( pos );
   }

   Item &at( int32 pos )
   {
      return m_itemarray.at( pos );
   }

   Item &operator[]( int32 pos ) throw()
   {
      return m_itemarray[pos];
   }

   const Item &operator[]( int32 pos ) const throw()
   {
      return m_itemarray[pos];
   }

   CoreObject *table() const { return m_table; }
   void table( CoreObject *t ) { m_table = t; }

   uint32 tablePos() const { return m_tablePos; }
   void tablePos( uint32 tp ) { m_tablePos = tp; }

   virtual void readProperty( const String &prop, Item &item );
   virtual void writeProperty( const String &prop, const Item &item );
   virtual void readIndex( const Item &pos, Item &target );
   virtual void writeIndex( const Item &pos, const Item &target );
};

}

#endif

/* end of flc_carray.h */

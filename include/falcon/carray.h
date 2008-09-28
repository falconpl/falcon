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

#define flc_ARRAY_GROWTH   32

namespace Falcon {

class Item;
class Bindings;

/** Core array (or array of items).
*/

class FALCON_DYN_CLASS CoreArray: public Garbageable
{
   uint32 m_alloc;
   uint32 m_size;
   Item *m_data;
   CoreDict *m_bindings;
   CoreObject *m_table;
   uint32 m_tablePos;

   CoreArray( VMachine *vm, Item *buffer, uint32 size, uint32 alloc );

public:

   /** Creates the core array. */
   CoreArray( VMachine *owner );
   CoreArray( VMachine *owner, uint32 prealloc );

   ~CoreArray();

   Item *elements() const { return m_data; }
   void elements( Item *d ) { m_data = d; }
   uint32 allocated() const { return m_alloc; }
   uint32 length()  const { return m_size; }
   void length( uint32 size ) { m_size = size; }
   void allocated( uint32 size ) { m_alloc = size; }

   void append( const Item &ndata );
   void prepend( const Item &ndata );
   void merge( const CoreArray &other );
   void merge_front( const CoreArray &other );

   bool insert( const Item &ndata, int32 pos );
   bool insert( const CoreArray &other, int32 pos );
   bool remove( int32 pos );
   bool remove( int32 first, int32 last );
   bool change( const CoreArray &other, int32 begin, int32 end );
   int32 find( const Item &itm ) const;
   bool insertSpace( uint32 pos, uint32 size );

   CoreArray *partition( int32 start, int32 end ) const;
   CoreArray *clone() const;

   void resize( uint32 size );
   void reserve( uint32 size );

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
   void setProperty( const String &name, Item &data );

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

   /** An inline utility to compute element size.
    *
    * @param count numbrer of elements
    * @return the amout of bytes needed to store the elements
    */
   int32 esize( int32 count=1 ) const { return sizeof( Item ) * count; }

   const Item &at( int32 pos ) const
   {
      if ( pos < 0 )
         pos = m_size + pos;
      if ( pos < 0 || pos > (int32) m_size )
         throw "Invalid range while accessing Falcon::CoreArray";
      return m_data[pos];
   }

   Item &at( int32 pos )
   {
      if ( pos < 0 )
         pos = m_size + pos;
      if ( pos < 0 || pos > (int32)m_size )
         throw "Invalid range while accessing Falcon::CoreArray";
      return m_data[pos];
   }

   Item &operator[]( int32 pos ) throw()
   {
      if ( pos < 0 )
         pos = m_size + pos;
      return m_data[pos];
   }

   const Item &operator[]( int32 pos ) const throw()
   {
      if ( pos < 0 )
         pos = m_size + pos;
      return m_data[pos];
   }

   CoreObject *table() const { return m_table; }
   void table( CoreObject *t ) { m_table = t; }

   uint32 tablePos() const { return m_tablePos; }
   void tablePos( uint32 tp ) { m_tablePos = tp; }

};

}

#endif

/* end of flc_carray.h */

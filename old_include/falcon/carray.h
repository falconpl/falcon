/*
   FALCON - The Falcon Programming Language.
   FILE: carray.h

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
#include <falcon/itemarray.h>
#include <falcon/callpoint.h>

#define flc_ARRAY_GROWTH   128

namespace Falcon {

class Item;
class Bindings;
class LinearDict;



/** Core array (or array of items).
*/

class FALCON_DYN_CLASS CoreArray: public DeepItem, public CallPoint
{
   ItemArray m_itemarray;
   CoreDict *m_bindings;
   CoreObject *m_table;
   /** Position in a table.
    *
    *  This indicator is also used to determine if an array can be a method.
    *
    *  Table arrays can never be methods (as they refer to the table they are
    *  stored in), so when m_table != 0, arrays are never methodic.
    *
    *  When m_table == 0, m_tablePos != 0 indicates a non-methodic array.
    */
   uint32 m_tablePos;

   CoreArray( Item *buffer, uint32 size, uint32 alloc );

public:

   /** Creates the core array. */
   CoreArray();
   CoreArray( const CoreArray& other );
   CoreArray( uint32 prealloc );

   ~CoreArray();

   virtual void gcMark( uint32 gen );

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

   virtual bool isFunc() const { return false; }
   virtual void readyFrame( VMachine* vm, uint32 paramCount );
   virtual const String& name() const;

   virtual void readProperty( const String &prop, Item &item );
   virtual void writeProperty( const String &prop, const Item &item );
   virtual void readIndex( const Item &pos, Item &target );
   virtual void writeIndex( const Item &pos, const Item &target );

   /** Determines if this array can be seen as a method in a class.
    *
    * If the array is part of a table, it can never be a method, and
    * this setting is ignored.
    *
    */
   void canBeMethod( bool b ) {
      if ( m_table == 0 )
         m_tablePos = b ? 0 : (uint32) -1;
   }

   /** Returns true if this array should be considered a method when callable and stored in a property.
    *
    */
   bool canBeMethod() const { return m_table == 0 && m_tablePos == 0; }
   
   /** Compare two arrays for deep equality.
      Internally calls item array's compare.
   */
   int compare( const CoreArray& other ) { return m_itemarray.compare( other.m_itemarray ); }
};

}

#endif

/* end of flc_carray.h */

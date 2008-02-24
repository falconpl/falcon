/*
   FALCON - The Falcon Programming Language.
   FILE: flc_carray.h
   $Id: carray.h,v 1.7 2007/08/11 00:11:51 jonnymind Exp $

   Core array
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

/** Core array (or array of items).

   Why not just use STL vectors?

   Because 1) this ones must be derived from garbageable, and 2) in future
   they may be polymorphic (we may have chuncked array).
*/

class FALCON_DYN_CLASS CoreArray: public Garbageable
{
   uint32 m_alloc;
   uint32 m_size;
   Item *m_data;

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
};

}

#endif

/* end of flc_carray.h */

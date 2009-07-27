/*
   FALCON - The Falcon Programming Language.
   FILE: itemarray.h

   Basic array of items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 27 Jul 2009 20:45:11 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_ITEMARRAY_H_
#define FALCON_ITEMARRAY_H_

#include <falcon/setup.h>
#include <falcon/sequence.h>

namespace Falcon
{

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

   /** Copy part or all of another vector on this vector.

       This method performs a flat copy of another array into this one.
       It is possible to copy part of the target array, specifying the first
       and amount parameters. Also, it's possible to select a starting position
       different from the beginning through the from parameter.

       If the from parameter is larger than the size of this array, or if the
       first parameter is larger than the size of the source array, the function
       returns false.

       If the required amount of items to be copied is greater than the number of
       items in the source array (starting from the first), then it's rounded down
       to copy all the items starting from first.

       If this array is not large enough to store all the items starting from the
       from parameter, it is enlarged.
       \param from The first item from which to start copying.
       \param src The source array
       \param first The first element in the array to be copied.
       \param amount Number of elements to be copied.
    */
   bool copyOnto( uint32 from, const ItemArray& src, uint32 first=0, uint32 amount=0xFFFFFFFF );

   /** Copy part or all of another vector on this vector.

       Shortcut for copyOnto starting from element 0 of this vector.
       \param src The source array
       \param first The first element in the array to be copied.
       \param amount Number of elements to be copied.
    */
   bool copyOnto( const ItemArray& src, uint32 first=0, uint32 amount=0xFFFFFFFF )
   {
      return copyOnto( 0, src, first, amount );
   }

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

}

#endif /* FALCON_ITEMARRAY_H_ */

/* end of itemarray.h */


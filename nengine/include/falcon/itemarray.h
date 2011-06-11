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
#include <falcon/types.h>
#include <falcon/item.h>

namespace Falcon
{

class Item;

class FALCON_DYN_CLASS ItemArray
{

public:
   ItemArray();
   ItemArray( const ItemArray& other );
   ItemArray( length_t prealloc );

   virtual ~ItemArray();

   Item *elements() const { return m_data; }
   void elements( Item *d ) { m_data = d; }
   length_t allocated() const { return m_alloc; }
   length_t length()  const { return m_size; }
   void length( length_t size ) { m_size = size; }
   void allocated( length_t size ) { m_alloc = size; }

   void clear() { m_size = 0; }
   bool empty() const { return m_size == 0; }

   /** Appends an item to the array.
      \param ndata The item to be added.
      \note The item in ndata will not have the Item::copied() bit set.
    */
   void append( const Item &ndata );

   /** Prepends an item to the array.
      \param ndata The item to be added.
      \note The item in ndata will not have the Item::copied() bit set.
    */
   void prepend( const Item &ndata );

   /** Apeends all the items in an array to this array.
      \param other The other array.
      \note All the merged items will have the Item::copied() bit set.
    */
   void merge( const ItemArray &other );

   /** Prepends all the items from an array to this array.
      \param other The other array.
      \note All the merged items will have the Item::copied() bit set.
    */
   void merge_front( const ItemArray &other );

   /** Inserts an item into the array.
      \param ndata The item to be added.
      \param pos The position at which the item will be inserted.
      \return false if \b pos is out of range.
    
    If \b pos is 0, the item will be inserted at the beginning of the
    array. If it's equal to length(), then the item will be appended at the end.

      \note The item in ndata will not have the Item::copied() bit set.
    */
   bool insert( const Item &ndata, length_t pos );
   
   /** Insert another array at a given position.
      \param other the other array from which items will be copied
      \param pos the position at which the array will be inserted.

    If \b pos is 0, the items will be inserted at the beginning of the
    target array. If it's equal to length(), then the items will be appended at the end.

      \note  All the merged items will not have the Item::copied() bit set.
    */
   bool insert( const ItemArray &other, length_t pos );
   bool remove( length_t pos );
   bool remove( length_t first, length_t count );
   bool change( const ItemArray &other, length_t begin, length_t count );
   int32 find( const Item &itm ) const;
   bool insertSpace( length_t pos, length_t size );

   void resize( length_t size );

   /* Returns a loop-free deep compare of the array. 
      The return value is the compare value of the first
      non-equal items, or -1 in case this array is the
      same but shorter than the other.
   */
   int compare( const ItemArray& other ) const { return compare( other, 0 ); }
   
   /**
    * Reduce the memory used by this array to exactly its size.
    */
   void compact();
   /** Reserve enough space as required.
    \param size the amount of space needed.
    */
   void reserve( length_t size );

   /** Generate a sub-array.
    \param start The first element to be taken.
    \param count Number of elements to take.
    \param bReverse if true, the returned partition will be reversed.
    \return a newly allocated ItemArray.
    */
   ItemArray *partition( length_t start, length_t count, bool bReverse = false ) const;

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
   bool copyOnto( length_t from, const ItemArray& src, length_t first=0, length_t amount=0xFFFFFFFF );

   /** Copy part or all of another vector on this vector.

       Shortcut for copyOnto starting from element 0 of this vector.
       \param src The source array
       \param first The first element in the array to be copied.
       \param amount Number of elements to be copied.
    */
   bool copyOnto( const ItemArray& src, length_t first=0, length_t amount=0xFFFFFFFF )
   {
      return copyOnto( 0, src, first, amount );
   }

   /** Returns the element at nth position. */
   inline const Item &at( length_t pos ) const
   {
      return m_data[pos];
   }

   inline Item &at( length_t pos )
   {
      return m_data[pos];
   }

   inline Item &operator[]( length_t pos )
   {
      return m_data[pos];
   }

   inline const Item &operator[]( length_t pos ) const
   {
      return m_data[pos];
   }


   /** An inline utility to compute element size.
    *
    * @param count numbrer of elements
    * @return the amout of bytes needed to store the elements
    */
   int32 esize( int32 count=1 ) const { return sizeof( Item ) * count; }


private:
   length_t m_alloc;
   length_t m_size;
   Item *m_data;
   length_t m_growth;

   ItemArray( Item *buffer, length_t size, length_t alloc );

   /** Classed used internally to track loops in traversals. */
   class Parentship
   {
   public:
      const ItemArray* m_array;
      Parentship* m_parent;

      Parentship( const ItemArray* d, Parentship* parent=0 ):
         m_array(d),
         m_parent( parent )
      {}
   };

   int compare( const ItemArray& other, Parentship* parent ) const;

   class Helper;
   friend class Helper;
};

}

#endif /* FALCON_ITEMARRAY_H_ */

/* end of itemarray.h */


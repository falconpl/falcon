/*
   FALCON - The Falcon Programming Language.
   FILE: itemreference.h

   Class implementing a reference to an item.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 03 Nov 2011 19:13:22 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_ITEMREFERENCE_H_
#define _FALCON_ITEMREFERENCE_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/item.h>

namespace Falcon {

/** Class implementing a reference to an item.
 
 This entity represents a garbage-collector sensible pointer to an item.
 */
class FALCON_DYN_CLASS ItemReference
{
public:
   
   ItemReference():
      m_mark(0)
   {}
      
   ItemReference( const Item& v ):
      m_item(v),
      m_mark(0)
   {}

   const Item& item() const { return m_item; }
   Item& item() { return m_item; }
   
   void gcMark( uint32 mark ) {
      if( m_mark != mark )
      {
         m_mark = mark;
         m_item.gcMark( mark );
      }
   }
   
   uint32 gcMark() const { return m_mark; }

   /** References an item, and turn it into a reference to us.
    \param item The item to be turned into a reference.
    Notice that the original item is changed into a reference to the item itself.
    */
   void reference( Item& target );
   
   /** Creates a newly allocated ItemReference to the target.
    \param target The target that will be trasformed into a reference to this item.
    */
   static ItemReference* create( Item& target ) 
   {
      ItemReference* ref = new ItemReference;
      ref->reference( target );
      return ref;
   }
   
private:
   Item m_item;
   uint32 m_mark;
};

}

#endif	/* FALCON_ITEMREFERENCE_H */

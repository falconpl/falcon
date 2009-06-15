/*
   FALCON - The Falcon Programming Language.
   FILE: deepitem.cpp

   Common interface for deep items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 02 Jan 2009 21:07:58 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_DEEP_ITEM_H
#define FLC_DEEP_ITEM_H

#include <falcon/setup.h>
#include <falcon/types.h>

namespace Falcon {

class String;
class Item;

/** Pure interface for deep item core data.
   Deep items are:
   - Strings
   - Object (instances)
   - Dictionaries
   - Arrays
   - Classes

   Code using this generic interface must be ready to
   receive Falcon::Error in throws and handle it correctly.

   This is always true for the VM and ancillary modules.

   Code directly implementing this access at lower level,
   as for example, getting a character from a string, doesn't
   need to be consistent with this behavior (i.e. it may just
   return false on set error).

   The rationale of this is that this is high-level code still
   working with abstract items, with no idea of the lower
   level implementation, even when this code is directly implemented
   into the target classes.
*/
class FALCON_DYN_CLASS DeepItem
{
public:
   virtual ~DeepItem() {}

   /** Implements the deep item interface.
      May throw  AccessError * if the property cannot be read.
   */
   virtual void readProperty( const String &, Item &item ) = 0;

   /** Implements the deep item interface.
      May throw AccessError * if the property cannot be set.
   */
   virtual void writeProperty( const String &, const Item &item ) = 0;

   /** Implements the deep item interface.
      May throw AccessError * if the pos doesn't represents a valid item
      index for the current type.
   */
   virtual void readIndex( const Item &pos, Item &target ) = 0;

   /** Implements the deep item interface.
      May throw AccessError * if the pos doesn't represents a valid item
      index for the current type or if the target item can't be set
      into this item at the given index.
   */
   virtual void writeIndex( const Item &pos, const Item &target ) = 0;
};

}

#endif

/* end of deepitem.h */


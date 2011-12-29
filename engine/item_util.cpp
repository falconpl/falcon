/*
   FALCON - The Falcon Programming Language.
   FILE: item_util.cpp

   Some utilities in the item class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 12 Jan 2011 20:37:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/item.h>
#include <falcon/itemarray.h>
#include <falcon/string.h>

#include "falcon/itemdict.h"

namespace Falcon {

int64 Item::len() const
{
   Class* cls;
   void* data;
   if( asClassInst( cls, data ) )
   {
       switch( cls->typeID() )
      {
      case FLC_CLASS_ID_STRING:
         return (int64) static_cast<String*>( data )->length();

      case FLC_CLASS_ID_ARRAY:
         return (int64) static_cast<ItemArray*>( data )->length();
      
      case FLC_CLASS_ID_DICT:
         return (int64) static_cast<ItemDict*>( data )->size();
      
      case FLC_CLASS_ID_RANGE:
         return (int64) 3;
            
      /* TODO MEMBUF*/

      default:
         return 0;
      }
   }
   else
   {
      return 0;
   }
}

}

/* end of item_util.cpp */

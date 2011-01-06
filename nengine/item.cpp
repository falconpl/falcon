/*
   FALCON - The Falcon Programming Language.
   FILE: item.cpp

   Item API implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ott 12 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Item API implementation
   This File contains the non-inlined access to items.
*/

#include <falcon/item.h>
#include <falcon/memory.h>
#include <falcon/mempool.h>
#include <falcon/common.h>
#include <falcon/symbol.h>
#include <falcon/coreobject.h>
#include <falcon/corefunc.h>
#include <falcon/carray.h>
#include <falcon/garbagepointer.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/membuf.h>
#include <falcon/vmmaps.h>
#include <falcon/error.h>
#include <cstdlib>
#include <cstring>


namespace Falcon
{

//===========================================================================
// Generic item manipulators

bool Item::isTrue() const
{
   switch( dereference()->type() )
   {
   case FLC_ITEM_NIL:
      return false;

   case FLC_ITEM_BOOL:
      return asBoolean() != 0;

   case FLC_ITEM_INT:
      return asInteger() != 0;

   case FLC_ITEM_NUM:
      return asNumeric() != 0.0;

   default:
      return false;
   }

   return false;
}

int64 Item::forceInteger() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();
   }
   return 0;
}


int64 Item::forceIntegerEx() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return asInteger();

      case FLC_ITEM_NUM:
         return (int64) asNumeric();

   }
   throw new TypeError( ErrorParam( e_param_type, __LINE__ ) );

   // to make some dumb compiler happy
   return 0;
}


numeric Item::forceNumeric() const
{
   switch( type() ) {
      case FLC_ITEM_INT:
         return (numeric) asInteger();

      case FLC_ITEM_NUM:
         return asNumeric();
   }
   return 0.0;
}

}

/* end of item.cpp */

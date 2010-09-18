/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_common.cpp

   Database Interface - common useful functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:25:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_common.h>
#include <falcon/vm.h>
#include <falcon/item.h>
#include <falcon/error.h>
#include <falcon/timestamp.h>

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

bool dbi_itemToSqlValue( const Item &item, String &value )
{
   switch( item.type() ) {
      case FLC_ITEM_BOOL:
         value = item.asBoolean() ? "TRUE" : "FALSE";
         return true;

      case FLC_ITEM_INT:
         value.writeNumber( item.asInteger() );
         return true;

      case FLC_ITEM_NUM:
         value.writeNumber( item.asNumeric(), "%f" );
         return true;

      case FLC_ITEM_STRING:
         dbi_escapeString( *item.asString(), value );
         value.prepend( "'" );
         value.append( "'" );
         return true;

      case FLC_ITEM_OBJECT: {
            CoreObject *o = item.asObject();
            //vm->itemToString( value, ??? )
            if ( o->derivedFrom( "TimeStamp" ) ) {
               TimeStamp *ts = (TimeStamp *) o->getUserData();
               ts->toString( value );
               value.prepend( "'" );
               value.append( "'" );
               return true;
            }
            return false;
         }

      case FLC_ITEM_NIL:
         value = "NULL";
         return true;

      default:
         return false;
   }
}


void dbi_escapeString( const String& input, String& value )
{
   uint32 len = input.length();
   uint32 pos = 0;
   value.reserve( len + 8 );

   while( pos < len )
   {
      uint32 chr = input.getCharAt(pos);
      switch( chr )
      {
         case '\\':
            value.append(chr);
            value.append(chr);
            break;

         case '\'':
            value.append( '\\' );
            value.append( '\'' );
            break;

         case '"':
            value.append( '\\' );
            value.append( '"' );
            break;

         default:
            value.append( chr );
      }
      ++pos;
   }
}


bool dbi_sqlExpand( const String& input, String& output, const ItemArray& arr )
{
   output.reserve( input.size() );
   output.size(0);
   String temp;

   uint32 iCount = 0;
   uint32 pos = 0;
   uint32 pos1 = input.find( "?" );

   while( pos1 != String::npos )
   {
      // too many ?
      if ( iCount >= arr.length() )
         return false;

      // can convert?
      if ( ! dbi_itemToSqlValue( arr[iCount++], temp ) )
         return false;

      // go!
      output += input.subString( pos, pos1 );
      output += temp;
      temp.size(0);

      // search next
      pos = pos1 + 1;
      pos1 = input.find( "?", pos );
   }

   // did we miss some elements in the array?
   if ( iCount != arr.length() )
      return false;

   output += input.subString( pos );
   return true;
}


}

/* end of dbi_mod.cpp */

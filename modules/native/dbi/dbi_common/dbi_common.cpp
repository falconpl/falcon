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
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

bool dbi_itemToSqlValue( const Item &item, String &value, char quote )
{
   switch( item.type() ) {
      case FLC_ITEM_NIL:
         value = "NULL";
         return true;

      case FLC_ITEM_BOOL:
         value = item.asBoolean() ? "TRUE" : "FALSE";
         return true;

      case FLC_ITEM_INT:
         value.writeNumber( item.asInteger() );
         return true;

      case FLC_ITEM_NUM:
         value.writeNumber( item.asNumeric(), "%f" );
         return true;

      case FLC_ITEM_USER:
         switch( item.asClass()->typeID() )
         {
         case FLC_CLASS_ID_STRING:
            dbi_escapeString( *item.asString(), value );
            return true;

         case FLC_CLASS_ID_TIMESTAMP:
            value.size(0);
            value.append(quote);
            TimeStamp* ts = static_cast<TimeStamp*>(item.asInst());
            ts->toString(value);
            value.append(quote);
            return true;
         }
         break;
   }

   return false;
}


void dbi_escapeString( const String& input, String& value, char quoteChr )
{
   uint32 len = input.length();
   uint32 pos = 0;
   value.reserve( len + 8 );
   value.append(quoteChr);

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
            break;
      }
      ++pos;
   }
   value.append(quoteChr);
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

/* end of dbi_common.cpp */

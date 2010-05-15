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

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

int dbi_itemToSqlValue( const Item &item, String &value )
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
         dbh_escapeString( *item.asString(), value );
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



void dbi_return_recordset( VMachine *vm, DBIRecordset *rec )
{
   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( rec );
   vm->retval( oth );
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

void dbi_throwError( const char* file, int line, int code, const String& desc )
{
   VMachine* vm = VMachine::getCurrent();

   if ( vm != 0 )
   {
      int msgId = code - FALCON_DBI_ERROR_BASE - 1;

      throw new DBIError( ErrorParam( code, line )
             .desc( vm->moduleString( msgId ) )
             .module( file )
             .extra( desc )
          );
   }
   else
   {
      throw new DBIError( ErrorParam( code, line )
         .desc( "Unknown error code" )
         .module( file )
         .extra( desc )
      );
   }
}


}

/* end of dbi_mod.cpp */

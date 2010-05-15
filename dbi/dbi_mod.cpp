/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_mod.cpp
 *
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Mon, 13 Apr 2009 18:56:48 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */


#include "dbi.h"
#include "dbi_mod.h"
#include "dbi_st.h"
#include <falcon/srv/dbi_service.h>

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

void dbh_addErrorDescription( VMachine *vm, Error* error )
{
   switch( error->errorCode() )
   {
   case FALCON_DBI_ERROR_COLUMN_RANGE:
      error->errorDescription( FAL_STR( dbi_msg_invalid_col ) );
      break;

   case FALCON_DBI_ERROR_INVALID_DRIVER:
      error->errorDescription( FAL_STR( dbi_msg_driver_not_found ) );
      break;

   case FALCON_DBI_ERROR_NOMEM:
      error->errorDescription( FAL_STR( dbi_msg_nomem ) );
      break;

   case FALCON_DBI_ERROR_CONNPARAMS:
      error->errorDescription( FAL_STR( dbi_msg_connparams ) );
      break;

   case FALCON_DBI_ERROR_CONNECT:
      error->errorDescription( FAL_STR( dbi_msg_connect ) );
      break;

   case FALCON_DBI_ERROR_QUERY:
      error->errorDescription( FAL_STR( dbi_msg_query ) );
      break;

   case FALCON_DBI_ERROR_QUERY_EMPTY:
      error->errorDescription( FAL_STR( dbi_msg_query_empty ) );
      break;

   case FALCON_DBI_ERROR_OPTPARAMS:
      error->errorDescription( FAL_STR( dbi_msg_option_error ) );
      break;

   case FALCON_DBI_ERROR_NO_SUBTRANS:
      error->errorDescription( FAL_STR( dbi_msg_no_subtrans ) );
      break;

   case FALCON_DBI_ERROR_NO_MULTITRANS:
      error->errorDescription( FAL_STR( dbi_msg_no_multitrans ) );
      break;

   }

}


int dbh_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value )
{
   switch( i->type() ) {
      case FLC_ITEM_BOOL:
         value = i->asBoolean() ? "TRUE" : "FALSE";
         return 1;

      case FLC_ITEM_INT:
         value.writeNumber( i->asInteger() );
         return 1;

      case FLC_ITEM_NUM:
         value.writeNumber( i->asNumeric(), "%f" );
         return 1;

      case FLC_ITEM_STRING:
         dbh_escapeString( *i->asString(), value );
         value.prepend( "'" );
         value.append( "'" );
         return 1;

      case FLC_ITEM_OBJECT: {
            CoreObject *o = i->asObject();
            //vm->itemToString( value, ??? )
            if ( o->derivedFrom( "TimeStamp" ) ) {
               TimeStamp *ts = (TimeStamp *) o->getUserData();
               ts->toString( value );
               value.prepend( "'" );
               value.append( "'" );
               return 1;
            }
            return 0;
         }

      case FLC_ITEM_NIL:
         value = "NULL";
         return 1;

      default:
         return 0;
   }
}



void dbh_return_recordset( VMachine *vm, DBIRecordset *rec )
{
   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( rec );
   vm->retval( oth );
}


void dbh_escapeString( const String& input, String& value )
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

void dbh_throwError( const char* file, int line, int code, const String& desc )
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

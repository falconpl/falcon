/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_ext.cpp
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun, 23 Dec 2007 22:02:37 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include <stdio.h>
#include <string.h>

#include <falcon/engine.h>
#include <falcon/error.h>

#include "dbi.h"
#include "dbi_ext.h"
#include "../include/dbiservice.h"

namespace Falcon {
namespace Ext {

CoreObject *dbi_defaultHandle; // Temporary until I figure how to set static class vars

/******************************************************************************
 * Local Helper Functions
 *****************************************************************************/

int DBIHandle_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value )
{
   if ( i->isInteger() ) {
      value.writeNumber( i->asInteger() );
      return 1;
   } else if ( i->isNumeric() ) {
      value.writeNumber( i->asNumeric(), "%f" );
      return 1;
   } else if ( i->isString() ) {
      dbh->escapeString( *i->asString(), value );
      value.prepend( "'" );
      value.append( "'" );
      return 1;
   } else if ( i->isObject() ) {
      CoreObject *o = i->asObject();
      //vm->itemToString( value, ??? )
      if ( o->derivedFrom( "TimeStamp" ) ) {
         TimeStamp *ts = (TimeStamp *) o->getUserData();
         ts->toString( value );
         value.prepend( "'" );
         value.append( "'" );
         return 1;
      }
   }

   return 0;
}

int DBIHandle_realSqlExpand( VMachine *vm, DBIHandle *dbh, String &sql, int startAt )
{
   char errorMessage[256];

   if ( vm->paramCount() > startAt )
   {
      // Check param 1, if a dict or object, we treat things special
      CoreDict *dict = NULL;
      CoreObject *obj = NULL;

      if ( vm->param( 1 )->isDict() )
         dict = vm->param( 1 )->asDict();
      else if ( vm->param( 1 )->isObject() )
         obj = vm->param( 1 )->asObject();

      uint32 dollarPos = sql.find( "$", 0 );

      while ( dollarPos != csh::npos ) {
         Item *i = NULL;
         int dollarSize = 1;

         if ( dollarPos == sql.length() - 1 ) {
            vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                        __LINE__ )
                                            .desc( "Stray $ charater at the end of query" ) ) );
            return 0;
         } else {
            if ( sql.getCharAt( dollarPos + 1 ) == '$' ) {
               sql.remove( dollarPos, 1 );
               dollarPos = sql.find( "$", dollarPos + 1 );
               continue;
            }

            if ( dict != NULL || obj != NULL) {
               uint32 commaPos = sql.find( ",", dollarPos );
               uint32 spacePos = sql.find( " ", dollarPos );
               uint32 ePos;
               if ( commaPos == csh::npos && spacePos == csh::npos )
                  ePos = sql.length();
               else if ( commaPos < spacePos )
                  ePos = commaPos;
               else
                  ePos = spacePos;

               if ( ePos == csh::npos ) {
                  String s( sql.subString( dollarPos ) );
                  s.prepend( "Failed to parse dollar expansion starting at: " );

                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error, __LINE__ )
                                                  .desc( s ) ) );
                  return 0;
               }

               String word = sql.subString( dollarPos + 1, ePos );
               if ( dict != NULL ) {
                  // Reading from the dict
                  Item wordI( &word );
                  i = dict->find( wordI );
               } else {
                  // Must be obj
                  i = obj->getProperty( word );
               }

               AutoCString asWord( word );
               if ( i == 0 ) {
                  if ( dict != NULL )
                     snprintf( errorMessage, 128, "Word expansion (%s) was not found in dictionary",
                              asWord.c_str() );
                  else
                     snprintf( errorMessage, 128, "Word expansion (%s) was not found in object",
                              asWord.c_str() );
      
                  GarbageString *s = new GarbageString( vm );
                  s->bufferize( errorMessage );
                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                              __LINE__ )
                                                  .desc( *s ) ) );
                  return 0;
               }

               dollarSize += word.length();

               // In case this fails a type check
               snprintf( errorMessage, 128, "Word expansion (%s) is an unknown type", asWord.c_str () );
            } else {
               AutoCString asTmp( sql.subString( dollarPos + 1 ) );
               int pIdx = atoi( asTmp.c_str() );

               if ( pIdx == 0 ) {
                  String s( sql.subString( dollarPos ) );
                  s.prepend( "Failed to parse dollar expansion starting at: " );

                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                              __LINE__ )
                                                  .desc( s ) ) );
                  return 0;
               }

               if ( pIdx > 99 ) dollarSize++; // it is 3 digits !?!?
               if ( pIdx > 9 ) dollarSize++;  // it is 2 digits
               dollarSize++;                  // it exists
               i = vm->param( pIdx + ( startAt - 1 ) );

               if ( i == 0 ) {
                  snprintf( errorMessage, 128, "Positional expansion (%i) is out of range", pIdx );
      
                  GarbageString *s = new GarbageString( vm );
                  s->bufferize( errorMessage );
                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                              __LINE__ )
                                                  .desc( *s ) ) );
                  return 0;
               }

               // In case this fails a type check
               snprintf( errorMessage, 128, "Positional expansion (%i) is an unknown type", pIdx );
            }
         }

         String value;
         if ( DBIHandle_itemToSqlValue( dbh, i, value ) == 0 ) {
            GarbageString *s = new GarbageString( vm );
            s->bufferize( errorMessage );

            vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_type_error, __LINE__ )
                                            .desc( *s ) ) );
            return 0;
         }

         sql.insert( dollarPos, dollarSize, value );

         dollarPos = sql.find( "$", dollarPos );
      }
   }

   return 1;
}

int DBIRecordset_getItem( VMachine *vm, DBIRecordset *dbr, dbi_type typ, int cIdx, Item &item )
{
   switch ( typ )
   {
   case dbit_string:
      {
         String value;
         dbi_status retval = dbr->asString( cIdx, value );
         switch ( retval )
         {
         case dbi_ok:
            {
               GarbageString *gsValue = new GarbageString( vm );
               gsValue->bufferize( value );

               item.setString( gsValue );
            }
            break;

         case dbi_nil_value:
            break;

         default:
            // TODO: handle error
            return 0;
         }
      }
      break;

   case dbit_integer:
      {
         int32 value;
         if ( dbr->asInteger( cIdx, value ) != dbi_nil_value )
            item.setInteger( (int64) value );
      }
      break;
   
   case dbit_integer64:
      {
         int64 value;
         if ( dbr->asInteger64( cIdx, value ) != dbi_nil_value )
            item.setInteger( value );
      }
      break;
   
   case dbit_numeric:
      {
         numeric value;
         if ( dbr->asNumeric( cIdx, value ) != dbi_nil_value )
            item.setNumeric( value );
      }
      break;
   
   case dbit_date:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDate( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;
   
   case dbit_time:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;
   
   case dbit_datetime:
      {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDateTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findGlobalItem( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

   default:
      return 0;
   }

   return 1;
}

int DBIRecordset_checkValidColumn( VMachine *vm, DBIRecordset *dbr, int cIdx )
{
   if ( cIdx >= dbr->getColumnCount() ) {
      char errorMessage[128];
      snprintf( errorMessage, 128, "Column index (%i) is out of range", cIdx );
      GarbageString *gs = new GarbageString( vm, errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( dbi_column_range_error, __LINE__ )
                                      .desc( *gs ) ) );
      return 0;
   } else if ( dbr->getRowIndex() == -1 ) {
      GarbageString *gs = new GarbageString( vm, "Invalid current row index" );
      vm->raiseModError( new DBIError( ErrorParam( dbi_row_index_invalid, __LINE__ )
                                      .desc( *gs ) ) );
      return 0;
   }

   return 1;
}

/******************************************************************************
 * Main DBIConnect
 *****************************************************************************/

FALCON_FUNC DBIConnect( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   if (  paramsI == 0 || ! paramsI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   String *params = paramsI->asString();
   String provName = *params;
   String connString = "";
   uint32 colonPos = params->find( ":" );

   if ( colonPos != csh::npos ) {
      provName = params->subString( 0, colonPos );
      connString = params->subString( colonPos + 1 );
   }

   DBIService *provider = theDBIService.loadDbProvider( vm, provName );
   if ( provider != 0 ) {
      // if it's 0, the service has already raised an error in the vm and we have nothing to do.
      String connectErrorMessage;
      dbi_status status;
      DBIHandle *hand = provider->connect( connString, false, status, connectErrorMessage );
      if ( hand == 0 ) {
         if ( connectErrorMessage.length() == 0 )
            connectErrorMessage = "An unknown error has occured during connect";

         vm->raiseModError( new DBIError( ErrorParam( status, __LINE__ )
                                          .desc( connectErrorMessage ) ) );

         return;
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = provider->makeInstance( vm, hand );
      vm->retval( instance );

      dbi_defaultHandle = instance;
   }

   // no matter what we return if we had an error.
}

/**********************************************************
   Handler class
 **********************************************************/

FALCON_FUNC DBIHandle_startTransaction( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   DBITransaction *trans = dbh->startTransaction();
   if ( trans == 0 ) {
      // raise an error depending on dbh->getLastError();
      return;
   }

   Item *trclass = vm->findGlobalItem( "%DBITransaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}

FALCON_FUNC DBIHandle_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   String sql( *sqlI->asString() );
   sql.bufferize();

   DBIHandle_realSqlExpand( vm, dbh, sql, 1 );

   dbi_status retval;
   DBIRecordset *recSet = dbh->query( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return;
   }

   Item *rsclass = vm->findGlobalItem( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( recSet );
   vm->retval( oth );
}

FALCON_FUNC DBIHandle_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   String sql( *sqlI->asString() );
   sql.bufferize();

   DBIHandle_realSqlExpand( vm, dbh, sql, 1 );

   dbi_status retval;
   int affectedRows = dbh->execute( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
   }

   vm->retval( affectedRows );
}

FALCON_FUNC DBIHandle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
}

FALCON_FUNC DBIHandle_getLastInsertedId( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   if ( vm->paramCount() == 0 )
      vm->retval( dbh->getLastInsertedId() );
   else {
      Item *sequenceNameI = vm->param( 0 );
      if ( sequenceNameI == 0 || ! sequenceNameI->isString() ) {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                           .origin( e_orig_runtime ) ) );
         return;
      }
      String sequenceName = *sequenceNameI->asString();
      vm->retval( dbh->getLastInsertedId( sequenceName ) );
   }
}

FALCON_FUNC DBIHandle_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String value;
   dbi_status retval = dbh->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( "Could not get last error message " ) ) );
      return;
   }

   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( value );

   vm->retval( gs );
}

FALCON_FUNC DBIHandle_sqlExpand( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   switch ( dbh->getQueryExpansionCapability() ) {
   case DBIHandle::q_dollar_sign_expansion:
      // TODO: Build array and ship off to query method
      return;

   case DBIHandle::q_question_mark_expansion:
      // TODO: Convert $1, $2 into ?, ? and ship off to query method
      return;

   default:
      // We will handle that below
      break;
   }

   Item *sqlI = vm->param( 0 );

   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   GarbageString *sql = new GarbageString( vm, *sqlI->asString() );
   if ( DBIHandle_realSqlExpand( vm, dbh, *sql, 1 ) )
      vm->retval( sql );
}

/**********************************************************
 * Transaction class
 **********************************************************/

FALCON_FUNC DBITransaction_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   String sql( *sqlI->asString() );
   sql.bufferize();

   DBIHandle_realSqlExpand( vm, dbt->getHandle(), sql, 1 );

   dbi_status retval;
   DBIRecordset *recSet = dbt->query( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return;
   }

   Item *rsclass = vm->findGlobalItem( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( recSet );
   vm->retval( oth );
}

FALCON_FUNC DBITransaction_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();

   Item *sqlI = vm->param( 0 );
   if ( sqlI == 0 || ! sqlI->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   String sql( *sqlI->asString() );
   sql.bufferize();

   DBIHandle_realSqlExpand( vm, dbt->getHandle(), sql, 1 );

   dbi_status retval;
   int affectedRows = dbt->execute( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
   }

   vm->retval( affectedRows );
}

FALCON_FUNC DBITransaction_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   dbt->close();

   vm->retval( 0 );
}

FALCON_FUNC DBITransaction_commit( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   dbi_status retval = dbt->commit();
   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( errorMessage ) ) );
      return;
   }

   vm->retval( 0 );
}

FALCON_FUNC DBITransaction_rollback( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   dbi_status retval = dbt->rollback();
   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( errorMessage ) ) );
      return;
   }

   vm->retval( 0 );
}
/******************************************************************************
 * Recordset class
 *****************************************************************************/

FALCON_FUNC DBIRecordset_next( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->next() );
}

FALCON_FUNC DBIRecordset_fetchArray( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      {
         String errorMessage;
         dbr->getLastError( errorMessage );

         vm->raiseModError( new DBIError( ErrorParam( nextRetVal, __LINE__ )
                                         .desc( errorMessage ) ) );
         return ;
      }
   }

   int cCount = dbr->getColumnCount();
   dbi_type cTypes[cCount];
   CoreArray *ary = new CoreArray( vm, cCount );
   
   dbr->getColumnTypes( cTypes );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) ) {
         ary->append( i );
      } else {
         // TODO: handle error
      }
   }

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_fetchDict( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      // TODO: Handle error
      break;
   }

   int cCount = dbr->getColumnCount();
   CoreDict *dict = new PageDict( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type cTypes[cCount];
   
   dbr->getColumnTypes( cTypes );
   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      dbi_status retval;
      GarbageString *gsName = new GarbageString( vm );
      gsName->bufferize( cNames[cIdx] );

      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         dict->insert( gsName, i );
      } else {
         // TODO: handle error
      }

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( dict );
}

FALCON_FUNC DBIRecordset_fetchObject( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *oI = vm->param( 0 );
   if ( oI == 0 || ! oI->isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .origin( e_orig_runtime ) ) );
      return;
   }

   CoreObject *o = oI->asObject();

   dbi_status nextRetVal = dbr->next();
   switch ( nextRetVal )
   {
   case dbi_ok:
      break;

   case dbi_eof:
      vm->retnil();
      return ;

   default:
      // TODO: Handle error
      break;
   }

   int cCount = dbr->getColumnCount();
   CoreDict *dict = new PageDict( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type cTypes[cCount];

   dbr->getColumnTypes( cTypes );
   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         o->setProperty( cNames[cIdx], i );
      } else {
         // TODO: handle error
      }

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( o );
}


FALCON_FUNC DBIRecordset_getRowCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getRowCount() );
}

FALCON_FUNC DBIRecordset_getColumnTypes( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   dbi_type cTypes[cCount];

   dbr->getColumnTypes( cTypes );

   for (int cIdx=0; cIdx < cCount; cIdx++ )
      ary->append( (int64) cTypes[cIdx] );

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_getColumnNames( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );

   dbr->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      GarbageString *gs = new GarbageString( vm );
      gs->bufferize( cNames[cIdx] );

      ary->append( gs );

      free( cNames[cIdx] );
   }

   free( cNames );

   vm->retval( ary );
}

FALCON_FUNC DBIRecordset_getColumnCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getColumnCount() );
}

FALCON_FUNC DBIRecordset_asString( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   String value;

   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   dbi_status retval = dbr->asString( cIdx, value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil();        // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asInteger( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   int32 value;

   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   dbi_status retval = dbr->asInteger( cIdx, value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil ();           // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asInteger64( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   int64 value;

   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   dbi_status retval = dbr->asInteger64( cIdx, value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil (); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asNumeric( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }

   numeric value;

   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   dbi_status retval = dbr->asNumeric( cIdx, value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asDate( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }


   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asDate( cIdx, *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }


   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asTime( cIdx, *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_asDateTime( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return;
   }


   int32 cIdx = columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   // create the timestamps
   TimeStamp *ts = new TimeStamp();
   Item *ts_class = vm->findGlobalItem( "TimeStamp" );
   //if we wrote the std module, can't be zero.
   fassert( ts_class != 0 );
   CoreObject *value = ts_class->asClass()->createInstance();
   dbi_status retval = dbr->asDateTime( cIdx, *ts );
   value->setUserData( ts );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil(); // TODO: handle the error
   else
      vm->retval( value );
}

FALCON_FUNC DBIRecordset_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   String value;
   dbi_status retval = dbr->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( "Could not get last error message " ) ) );
      return;
   }

   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( value );

   vm->retval( gs );

   vm->retval( 0 );
}

FALCON_FUNC DBIRecordset_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   dbr->close();
}

//======================================================
// DBI Record
//======================================================

FALCON_FUNC DBIRecord_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   int paramCount = vm->paramCount();

   // Populate tableName, primaryKey and persist
   if ( paramCount > 0 )
      einst->setProperty( "_tableName",  *vm->param( 0 ) );
   if ( paramCount > 1 )
      einst->setProperty( "_primaryKey", *vm->param( 1 ) );
   if ( paramCount > 2 )
      einst->setProperty( "_persist",    *vm->param( 2 ) );
   einst->setProperty( "_dbh", Item( dbi_defaultHandle ) );
}

int DBIRecord_getPersistPropertyNames( VMachine *vm, CoreObject *self, String columnNames[], int maxColumnCount )
{
   Item *persistI = self->getProperty( "_persist" );

   if ( persistI == 0 || persistI->isNil() ) {
      // No _persist, loop through all public properties
      int pCount = self->propCount();
      int cIdx = 0;

      for ( int pIdx=0; pIdx < pCount; pIdx++ ) {
         String p = self->getPropertyName( pIdx );
         if ( p.getCharAt( 0 ) != '_' ) {
            Item i = self->getPropertyAt( pIdx );
            if ( i.isInteger() || i.isNumeric() || i.isObject() || i.isString() ) {
               columnNames[cIdx] = p;
               cIdx++;
            }
         }
      }

      return cIdx;
   } else if ( ! persistI->isArray() ) {
      // They gave a _persist property, but it's not an array
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .origin( e_orig_runtime ) ) );
      return 0;
   } else {
      // They gave a _persist property, trust it
      CoreArray *persist = persistI->asArray();
      int cCount = persist->length();

      for ( int cIdx=0; cIdx < cCount; cIdx++) {
          columnNames[cIdx] = *persist->at( cIdx ).asString();
      }

      return cCount;
   }
}

FALCON_FUNC DBIRecord_insert( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *tableNameI = self->getProperty( "_tableName" );
   Item *primaryKeyI = self->getProperty( "_primaryKey" );
   Item *dbhI = self->getProperty( "_dbh" );

   CoreObject *dbhO = dbhI->asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI->asString();
   String *primaryKey = primaryKeyI->asString();

   int propertyCount = self->propCount();
   String *columnNames = new String[propertyCount];

   propertyCount = DBIRecord_getPersistPropertyNames( vm, self, columnNames, propertyCount );

   String sql;
   sql.append( "INSERT INTO " );
   sql.append( *tableName );
   sql.append( "( " );

   for ( int cIdx=0; cIdx < propertyCount; cIdx++ ) {
      if ( cIdx > 0 )
         sql.append( ", " );
      sql.append( columnNames[cIdx] );
   }

   sql.append( " ) VALUES ( " );

   for ( int cIdx=0; cIdx < propertyCount; cIdx++ ) {
      if ( cIdx > 0 )
         sql.append( ", " );

      String value;
      if ( DBIHandle_itemToSqlValue( dbh, self->getProperty( columnNames[cIdx] ), value ) == 0 ) {
         String errorMessage = "Invalid type for ";
         errorMessage.append( columnNames[cIdx] );

         vm->raiseModError( new DBIError( ErrorParam( dbi_invalid_type, __LINE__ )
                                         .desc( errorMessage ) ) );
         return;
      }

      sql.append( value );
   }

   sql.append( " )" );

   vm->retval( new GarbageString( vm, sql ) );
}

FALCON_FUNC DBIRecord_update( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *tableNameI = self->getProperty( "_tableName" );
   Item *primaryKeyI = self->getProperty( "_primaryKey" );
   Item *dbhI = self->getProperty( "_dbh" );

   CoreObject *dbhO = dbhI->asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI->asString();
   String *primaryKey = primaryKeyI->asString();

   int propertyCount = self->propCount();
   String *columnNames = new String[propertyCount];

   propertyCount = DBIRecord_getPersistPropertyNames( vm, self, columnNames, propertyCount );

   String sql;

   sql.append( "UPDATE " );
   sql.append( *tableName );
   sql.append( " SET " );

   for ( int cIdx=0; cIdx < propertyCount; cIdx++ ) {
      if ( cIdx > 0 )
         sql.append( ", " );
      Item *i = self->getProperty( columnNames[cIdx] );

      String value;
      if ( DBIHandle_itemToSqlValue( dbh, i, value ) == 0 ) {
         String errorMessage = "Invalid type for ";
         errorMessage.append( columnNames[cIdx] );

         vm->raiseModError( new DBIError( ErrorParam( dbi_invalid_type, __LINE__ )
                                         .desc( errorMessage ) ) );
         return;
      }

      sql.append( columnNames[cIdx] );
      sql.append( " = " );
      sql.append( value );
   }

   Item *primaryKeyValueI = self->getProperty( *primaryKey );
   String value;
   if ( DBIHandle_itemToSqlValue( dbh, primaryKeyValueI, value ) == 0 ) {
      String errorMessage = "Invalid type for primary key ";
      errorMessage.append( *primaryKey );

      vm->raiseModError( new DBIError( ErrorParam( dbi_invalid_type, __LINE__ )
                                      .desc( errorMessage ) ) );
      return;
   }

   sql.append( " WHERE " );
   sql.append( *primaryKey );
   sql.append( " = " );
   sql.append( value );

   vm->retval( new GarbageString( vm, sql ) );
}

FALCON_FUNC DBIRecord_delete( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *tableNameI = self->getProperty( "_tableName" );
   Item *primaryKeyI = self->getProperty( "_primaryKey" );
   Item *dbhI = self->getProperty( "_dbh" );

   CoreObject *dbhO = dbhI->asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI->asString();
   String *primaryKey = primaryKeyI->asString();
   Item *pkValueI = self->getProperty( *primaryKey );
   String value;

   DBIHandle_itemToSqlValue( dbh, pkValueI, value );

   String sql = "DELETE FROM ";
   sql.append( *tableName );
   sql.append( " WHERE " );
   sql.append( *primaryKey );
   sql.append( " = " );
   sql.append( value );

   vm->retval( new GarbageString( vm, sql ) );
}

//======================================================
// DBI error
//======================================================

FALCON_FUNC DBIError_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new DBIError ) );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of dbi_ext.cpp */


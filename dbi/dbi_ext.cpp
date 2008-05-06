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
 * (C) Copyright 2007,2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <stdio.h>
#include <string.h>

#include <falcon/engine.h>
#include <falcon/error.h>

#include "dbi.h"
#include "dbi_ext.h"
#include "../include/dbiservice.h"

/*#
   @beginmodule dbi
*/

namespace Falcon {
namespace Ext {

CoreObject *dbi_defaultHandle; // Temporary until I figure how to set static class vars

/******************************************************************************
 * Local Helper Functions
 *****************************************************************************/

static int DBIHandle_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value )
{
   switch( i->type() ) {
      case FLC_ITEM_BOOL:
         value = i->asBoolean() ? "'TRUE'" : "'FALSE'";
         return 1;

      case FLC_ITEM_INT:
         value.writeNumber( i->asInteger() );
         return 1;

      case FLC_ITEM_NUM:
         value.writeNumber( i->asNumeric(), "%f" );
         return 1;

      case FLC_ITEM_STRING:
         dbh->escapeString( *i->asString(), value );
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

static int DBIHandle_realSqlExpand( VMachine *vm, DBIHandle *dbh, String &sql, int startAt=0 )
{
   String errorMessage;

   Item *sqlI = vm->param( startAt );
   if ( sqlI == 0 || ! sqlI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") ) );
      return 0;
   }

   sql = *sqlI->asString();
   sql.bufferize();

   startAt++;

   uint32 dollarPos = sql.find( "$", 0 );

   if ( dollarPos != csh::npos )
   {
      // Check param 'startAt', if a dict or object, we treat things special
      CoreDict *dict = NULL;
      CoreObject *obj = NULL;

      Item *psI = vm->param( startAt );
      if ( psI != 0 ) {
         if ( psI->isDict() )
            dict = psI->asDict();
         else if ( psI->isObject() )
            obj = psI->asObject();
      }

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
                  String s = "starting at: " + sql.subString( dollarPos );

                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error, __LINE__ )
                                                  .desc( "Failed to parse dollar expansion" )
                                                  .extra( s ) ) );
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

               if ( i == 0 ) {
                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error, __LINE__ )
                                                  .desc( "Word expansion was not found in dictionary/object" )
                                                  .extra( word ) ) );
                  return 0;
               }

               dollarSize += word.length();
            } else {
               AutoCString asTmp( sql.subString( dollarPos + 1 ) );
               int pIdx = atoi( asTmp.c_str() );

               if ( pIdx == 0 ) {
                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error,
                                                              __LINE__ )
                                                  .desc( "Failed to parse dollar expansion" )
                                                  .extra( "from: " + sql.subString( dollarPos ) ) ) );
                  return 0;
               }

               if ( pIdx > 99 ) dollarSize++; // it is 3 digits !?!?
               if ( pIdx > 9 ) dollarSize++;  // it is 2 digits
               dollarSize++;                  // it exists
               i = vm->param( pIdx + ( startAt - 1 ) );

               errorMessage.writeNumber( (int64) pIdx );

               if ( i == 0 ) {
                  vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_error, __LINE__ )
                                                  .desc("Positional expansion out of range")
                                                  .extra( errorMessage ) ) );
                  return 0;
               }
            }
         }

         String value;
         if ( DBIHandle_itemToSqlValue( dbh, i, value ) == 0 ) {
            vm->raiseModError( new DBIError( ErrorParam( dbi_sql_expand_type_error, __LINE__ )
                     .desc( "Failed to expand a value due to it being an unknown type" )
                     .extra( "from: " + sql.subString( dollarPos ) ) ) );
            return 0;
         }

         sql.insert( dollarPos, dollarSize, value );
         dollarPos = sql.find( "$", dollarPos );
      }
   }

   return 1;
}

static int DBIRecordset_getItem( VMachine *vm, DBIRecordset *dbr, dbi_type typ, int cIdx, Item &item )
{
   switch ( typ )
   {
      case dbit_string: {
         String value;
         dbi_status retval = dbr->asString( cIdx, value );
         switch ( retval )
         {
            case dbi_ok: {
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

      case dbit_integer: {
         int32 value;
         if ( dbr->asInteger( cIdx, value ) != dbi_nil_value )
            item.setInteger( (int64) value );
      }
      break;

      case dbit_boolean: {
         bool value;
         if ( dbr->asBoolean( cIdx, value ) != dbi_nil_value ) {
            item.setBoolean( value );
         }
      }

      case dbit_integer64: {
         int64 value;
         if ( dbr->asInteger64( cIdx, value ) != dbi_nil_value )
            item.setInteger( value );
      }
      break;

      case dbit_numeric: {
         numeric value;
         if ( dbr->asNumeric( cIdx, value ) != dbi_nil_value )
            item.setNumeric( value );
      }
      break;

      case dbit_date: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDate( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

      case dbit_time: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
            fassert( ts_class != 0 );
            CoreObject *value = ts_class->asClass()->createInstance();
            value->setUserData( ts );
            item.setObject( value );
         }
      }
      break;

      case dbit_datetime: {
         TimeStamp *ts = new TimeStamp();
         if ( dbr->asDateTime( cIdx, *ts ) != dbi_nil_value ) {
            Item *ts_class = vm->findWKI( "TimeStamp" );
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

static int DBIRecordset_checkValidColumn( VMachine *vm, DBIRecordset *dbr, int cIdx )
{
   if ( cIdx >= dbr->getColumnCount() ) {
      String errorMessage = "Column index (";
      errorMessage.writeNumber( (int64) cIdx );
      errorMessage += ") is out of range";

      vm->raiseModError( new DBIError( ErrorParam( dbi_column_range_error, __LINE__ )
                                      .desc( errorMessage ) ) );
      return 0;
   } else if ( dbr->getRowIndex() == -1 ) {
      vm->raiseModError( new DBIError( ErrorParam( dbi_row_index_invalid, __LINE__ )
                                      .desc( "Invalid current row index" ) ) );
      return 0;
   }

   return 1;
}

static int DBIHandle_realExecute( VMachine *vm, DBIHandle *dbh, const String &sql )
{
   dbi_status retval;
   int affectedRows = dbh->execute( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( errorMessage ) ) );
      return -1;
   }

   return affectedRows;
}

static void DBIRecord_execute( VMachine *vm, DBIHandle *dbh, const String &sql )
{
   dbi_status retval;
   int affectedRows;

   if ( vm->paramCount() == 0 )
      affectedRows = dbh->execute( sql, retval );
   else {
      Item *trI = vm->param( 0 );
      if ( trI == 0 || ! trI->isObject() || ! trI->asObject()->derivedFrom( "DBITransaction" ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "DBITransaction" ) ) );
         return;
      }
      CoreObject *trO = trI->asObject();
      DBITransaction *tr = static_cast<DBITransaction *>( trO->getUserData() );
      affectedRows = tr->execute( sql, retval );
   }

   if ( retval == dbi_ok )
      vm->retval( affectedRows );
   else {
      String errorMessage;
      dbh->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
               .desc( errorMessage ) ) );
   }
}

static int DBIRecord_getPersistPropertyNames( VMachine *vm, CoreObject *self, String columnNames[], int maxColumnCount )
{
   Item *persistI = self->getProperty( "_persist" );

   if ( persistI == 0 || persistI->isNil() ) {
      // No _persist, loop through all public properties
      int pCount = self->propCount();
      int cIdx = 0;

      for ( int pIdx=0; pIdx < pCount; pIdx++ ) {
         const String &p = self->getPropertyName( pIdx );
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
                                         .extra( "_persist.type()!=A" ) ) );
      return 0;
   } else {
      // They gave a _persist property, trust it
      CoreArray *persist = persistI->asArray();
      int cCount = persist->length();

      for ( int cIdx=0; cIdx < cCount; cIdx++) {
         const Item &pi = persist->at( cIdx );
         if ( ! pi.isString() )
         {
            vm->raiseModError( new DBIError( ErrorParam( dbi_row_index_invalid, __LINE__ )
                                            .desc( "There was a non-string item in the \"_persist\" property" ) ) );
            return 0;
         }
         else
            columnNames[cIdx] = *pi.asString();
      }

      return cCount;
   }
}

static DBIRecordset *DBIHandle_baseQueryOne( VMachine *vm, int startAt = 0 )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String sql;
   DBIHandle_realSqlExpand( vm, dbh, sql, startAt );

   dbi_status retval;
   DBIRecordset *recSet = dbh->query( sql, retval );
   if ( recSet == NULL ) {
      vm->retnil();
      return NULL;
   }

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return NULL;
   }

   dbi_status nextStatus = recSet->next();
   if ( nextStatus != dbi_ok ) {
      vm->retnil();
      return NULL;
   }

   return recSet;
}

/******************************************************************************
 * Main DBIConnect
 *****************************************************************************/

/*#
 @function DBIConnect
 @brief Connect to a database server.
 @param String SQL connection string.
 @return an instance of @a DBIHandle.

 Known connection strings are:
  - <code>pgsql:normal postgresql connection string</code>
  - <code>sqlite3:sqlite_db_filename.db</code>
  - <code>mysql:normal mysql connection string</code>
 */

FALCON_FUNC DBIConnect( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   if (  paramsI == 0 || ! paramsI->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S" ) ) );
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
                                          .desc( "Uknown error (**)" )
                                          .extra( connectErrorMessage ) ) );

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

/*#
 @method startTransaction DBIHandle
 @brief Start a transaction
 @return an instance of @a DBITransaction

 This method returns a new transaction.
 */

FALCON_FUNC DBIHandle_startTransaction( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   DBITransaction *trans = dbh->startTransaction();
   if ( trans == NULL ) {
      String errorMessage;
      dbh->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( dbi_error, __LINE__ )
                                      .desc( errorMessage ) ) );
      return;
   }

   Item *trclass = vm->findWKI( "%DBITransaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}

/*#
 @method query DBIHandle
 @brief Execute a SQL query that expects to have data as a result
 @param String SQL query
 @return an instance of @a DBIRecordset
 */

FALCON_FUNC DBIHandle_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String sql;
   if ( DBIHandle_realSqlExpand( vm, dbh, sql, 0 ) == 0 )
      return;

   dbi_status retval;
   DBIRecordset *recSet = dbh->query( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbh->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return;
   }

   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( recSet );
   vm->retval( oth );
}

dbi_type *DBIHandle_getTypes( DBIRecordset *recSet )
{
   if (recSet == NULL )
      return NULL;
   dbi_type *cTypes = (dbi_type *) malloc( sizeof( dbi_type ) * recSet->getColumnCount() );
   recSet->getColumnTypes( cTypes );
   return cTypes;
}

/*#
 @method queryOne DBIHandle
 @brief Perform the SQL query and return the first field of the first record.
 @param String SQL query
 @return value or nil if no results found.

 @see DBIHandle.queryOneArray
 @see DBIHandle.queryOneDict
 @see DBIHandle.queryOneObject
 */

FALCON_FUNC DBIHandle_queryOne( VMachine *vm )
{
   DBIRecordset *recSet = DBIHandle_baseQueryOne( vm );
   if ( recSet == NULL ) {
      vm->retnil();
      return;
   }
   dbi_type *cTypes = DBIHandle_getTypes( recSet );

   Item i;
   int32 id;
   recSet->asInteger( 0, id );
   if ( DBIRecordset_getItem( vm, recSet, cTypes[0], 0, i ) )
      vm->retval( i );
   recSet->close();

   free( cTypes );
}

/*#
 @method queryOneArray DBIHandle
 @brief Perform the SQL query and return only the first record as an array.
 @param String SQL query
 @return Array populated array on nil on no results found.

 @see DBIHandle.queryOne
 @see DBIHandle.queryOneDict
 @see DBIHandle.queryOneObject
 */

FALCON_FUNC DBIHandle_queryOneArray( VMachine *vm )
{
   DBIRecordset *recSet = DBIHandle_baseQueryOne( vm );
   if ( recSet == NULL )
      return; // TODO: thrown an error

   int cCount = recSet->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   dbi_type *cTypes = DBIHandle_getTypes( recSet );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 ) {
         free( cTypes );
         return;
      }
      ary->append( i );
   }
   vm->retval( ary );
   free( cTypes );
}

/*#
 @method queryOneDict DBIHandle
 @brief Perform the SQL query and return only the first record as a Dictionary.
 @param String SQL query
 @return Dictionary populated with result or nil on no results found.

 @see DBIHandle.queryOne
 @see DBIHandle.queryOneArray
 @see DBIHandle.queryOneObject
 */

FALCON_FUNC DBIHandle_queryOneDict( VMachine *vm )
{
   DBIRecordset *recSet = DBIHandle_baseQueryOne( vm );
   if ( recSet == NULL )
      return; // TODO: thrown an error

   int cCount = recSet->getColumnCount();
   PageDict *dict = new PageDict( vm, cCount );
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = DBIHandle_getTypes( recSet );

   recSet->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 ) {
         free( cTypes );
         free( cNames );
         return;
      }

      GarbageString *gsName = new GarbageString( vm );
      gsName->bufferize( cNames[cIdx] );

      dict->insert( gsName, i );
   }

   free( cTypes );
   free( cNames );

   vm->retval( dict );
}

/*#
 @method queryOneObject DBIHandle
 @brief Perform the SQL query and return only the first record as an Object.
 @param Object object to populate
 @param String SQL query
 @return populated object on success, or nil on no results found.

 Refer to documentation on @a DBIRecord for more information on using the DBI object
 query system.

 @see DBIHandle.queryOne
 @see DBIHandle.queryOneArray
 @see DBIHandle.queryOneDict
 */

FALCON_FUNC DBIHandle_queryOneObject( VMachine *vm )
{
   Item *objI = vm->param( 0 );
   if ( objI == 0 || ! objI->isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "O" ) ) );
      return;
   }

   CoreObject *obj = objI->asObject();
   DBIRecordset *recSet = DBIHandle_baseQueryOne( vm, 1);
   if (recSet == NULL) {
      vm->retnil();
      return; // TODO: Return error
   }

   int cCount = recSet->getColumnCount();
   //TODO: Bufferize this things.
   char **cNames = (char **) malloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = DBIHandle_getTypes( recSet );

   recSet->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 ) {
         String indexString;
         indexString.writeNumber( (int64) cIdx );

         vm->raiseModError( new DBIError( ErrorParam( 0, __LINE__ )
                                         .desc( "Could not retrieve column value" )
                                         .extra( indexString ) ) );

         free( cTypes );
         free( cNames );
         return;
      }

      obj->setProperty( cNames[cIdx], i );
   }

   free( cTypes );
   free( cNames );

   vm->retval( obj );
}

/*#
 @method execute DBIHandle
 @brief Execute the SQL statement.
 @param String SQL statment
 @return number of affected rows

 Used for SQL queries that do not expect a resultset in return.

 @see DBIHandle.query
 */

FALCON_FUNC DBIHandle_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String sql;
   DBIHandle_realSqlExpand( vm, dbh, sql );

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

/*#
 @method close DBIHandle
 @brief Close the database handle.
 */

FALCON_FUNC DBIHandle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
}

/*#
 @method getLastInsertedId DBIHandle
 @brief Get the ID of the last record inserted.
 @return Integer

 This is database dependent but so widely used, it is included in the DBI module. Some
 databases such as MySQL only support getting the last inserted ID globally in the
 database server while others like PostgreSQL allow you to get the last inserted ID of
 any table. Thus, it is suggested that you always supply the sequence id as which to
 query. DBI drivers such as MySQL are programmed to ignore the extra information and
 return simply the last ID inserted into the database.
 */

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
                                           .extra( "S" ) ) );
         return;
      }
      String sequenceName = *sequenceNameI->asString();
      vm->retval( dbh->getLastInsertedId( sequenceName ) );
   }
}

/*#
 @method getLastError DBIHandle
 @brief Get the last error string from the database server.
 @return String

 This string is database server dependent. It is provided to get detailed information
 as to the error.
 */

FALCON_FUNC DBIHandle_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   String value;
   dbi_status retval = dbh->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( "Unknown error" )
                                      .extra( "Could not get last error message " ) ) );
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

   String sql;
   if ( DBIHandle_realSqlExpand( vm, dbh, sql, 0 ) )
      vm->retval( new GarbageString( vm , sql ) );
}

/**********************************************************
 * Transaction class
 **********************************************************/

/*#
 @method query DBITransaction
 @brief Perform a query that returns row data as part of the transaction.
 @param String SQL query
 @return @a DBIRecordset instance

 A failed query will cause the transaction to fail as well.
 */

FALCON_FUNC DBITransaction_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   String sql;
   if ( DBIHandle_realSqlExpand( vm, dbt->getHandle(), sql ) == 0 )
      return;

   dbi_status retval;
   DBIRecordset *recSet = dbt->query( sql, retval );

   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );

      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                       .desc( errorMessage ) ) );
      return;
   }

   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( recSet );
   vm->retval( oth );
}

/*#
 @method execute DBITransaction
 @brief Perform a query that does not expect row data as a result, as part of this
 transaction.
 @param String SQL execute statement
 @return number of affected rows

 A failed execute will cause the transaction to fail as well.
 */

FALCON_FUNC DBITransaction_execute( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   String sql;
   if ( DBIHandle_realSqlExpand( vm, dbt->getHandle(), sql ) == 0 )
      return;

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

/*#
 @method close DBITransaction
 @brief Close the transaction automatically committing or rolling back the transaction.
 @return 1 on success

 As to a commit or rollback, it depends on the current transaction status.
 */

FALCON_FUNC DBITransaction_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   dbt->close();
}

/*#
 @method commit DBITransaction
 @brief Commit the transaction to the database.
 @return 1 on success

 This does not close the transaction. You can perform a commit at safe steps within
 the transaction if necessary.
 */

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
}

/*#
 @method rollback DBITransaction
 @brief Rollback the transaction (undo) to last commit point.
 @return 1 on success

 This does not close the transaction. You can rollback and try another operation
 within the same transaction as many times as you wish.
 */

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
}

/*#
   @method openBlob DBITransaction
   @param Opens an existing blob entity.
   @param blobID A string containing the blob ID to be opened.
   @return On success, a DBIBlobStream that can be used to read or write from/to the blob.
   @raise DBIError on error.

   This method allows to open a stream towards a blob object. The returned stream
   can be manipulated as any other stream; it can be seeked, it can be truncated,
   it can be read and written both binary and text oriented.

   If the amount of data to be written or read is limited, or if many blobs must
   be read or written in row, prefer the @a DBITransaction.readBlob and
   @a DBITransaction.createBlob methods.

   Drivers may return instances of the DBIBlobStream class, or provide their own
   specific sublcasses instead.
*/
FALCON_FUNC DBITransaction_openBlob( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   Item *i_blobID = vm->param( 0 );
   if ( i_blobID == 0 || ! i_blobID->isString() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") ) );
      return;
   }

   dbi_status retval;
   DBIBlobStream *stream = dbt->openBlob( *i_blobID->asString(), retval );
   if ( stream == 0 ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
                                      .desc( errorMessage ) ) );
      return;
   }

   // create the class suggested by the stream itself
   Item *blobstream_class = vm->findWKI( stream->getFalconClassName() );
   // if the driver did his things, the blob stream subclass should exist.
   fassert( blobstream_class != 0 );

   CoreObject *objStream = blobstream_class->asClass()->createInstance();
   // then apply the class to its own stream
   objStream->setUserData( stream );
   // and return it
   vm->retval( objStream );
}

/*#
   @method createBlob DBITransaction
   @brief Creates a new blob entity.
   @optparam data A string or a membuffer to be written.
   @optparam options A string containing driver specific blob creation parameters.
   @return A stream that can be used to read or write from/to the blob,
           or just the blob ID.
   @raise DBIError on error.

   This method creates a Blob entity in the database. If a data member is not given,
   then a @a DBIBlobStream class is returned; the returned instance can be used to
   fill the blob with data and to retreive informations about the blob entity.

   If @b data member is given, it must be either a string or a membuf, and it will be
   used to fill the blob entity. If a string is given, it will be written through the
   string oriented Stream.writeText method, while if it's a membuf, it will be written
   through the byte oriented Stream.write method (although the underlying driver may
   just treat them the same, or rather use the @b options parameter to decide how
   to write the data). After the data is written, the created blob is closed and its
   ID is returned instead of the blob stream.

   The @b options parameter can be used to send driver specific options that may control
   the type and creation parameters of the blog entity. Please, refer to the specific
   DBI driver documentation for details.

   Drivers may return instances of the DBIBlobStream class, or provide their own
   specific sublcasses instead.
*/
FALCON_FUNC DBITransaction_createBlob( VMachine *vm )
{
}

/*#
   @method readBlob DBITransaction
   @brief Reads a whole blob entity.
   @param blobId The ID of the blob entity to be read.
   @optparam data A string or a MemBuf to be read.
   @optparam maxlen Maximum length read from the blob.
   @return A string (or a MemBuf, if that was used as a parmeter) containing the whole blob data.
   @raise DBIError on error or if the ID is invalid.

   This method reads a whole blob in memory. If the parameter @b data is not given, then a new
   string, long enough to store all the blob, will be created. If it's given and it's a string,
   then the string buffer will be used, and it will be eventually expanded if needed. If it's
   a MemBuf, then the method will read at maximum the size of the MemBuffer in bytes.

   The maximum size of the input may be also limited by the @b maxlen parameter. If both
   @b maxlen and @b data (as a MemBuf) are provided, the maximum size actually read will be
   the minimum of @b maxlen, the @b data MemBuf size and the blob entity size.

   @note It is possible to give a maximum length to be read and create dynamically the needed
   space by setting @b data to nil and passing @b maxlen.
*/
FALCON_FUNC DBITransaction_readBlob( VMachine *vm )
{
}

/*#
   @method write DBITransaction
   @brief Overwrites an existing blob entity.
   @param blobId The ID of the blob entity to be overwritten.
   @param data A string or a MemBuf to be written.
   @optparam start Character (if data is string) or byte (if data is MemBuf) from which to
      start writing.
   @optparam length Maximum count of  Characters (if @b data is a string) or bytes (if @b data a is MemBuf)
         to be written.
   @return A string (or a MemBuf, if that was used as a parmeter) containing the whole blob data.
   @raise DBIError on error or if the ID is invalid.

   This method overwrites a whole existing blob entity. The selection range allows to
   write a portion of the existing data obviating the need to extract a subpart of it.
*/
FALCON_FUNC DBITransaction_writeBlob( VMachine *vm )
{
}



/******************************************************************************
 * Recordset class
 *****************************************************************************/

/*#
 @method next DBIRecordset
 @brief Advanced the record pointer to the next record.
 @return 1 if successful, 0 if not successful, usually meaning the EOF has been
   hit.

 All new queries are positioned before the first record, meaning, next should be
 called before accessing any values.

 @see DBIRecordset.fetchArray
 @see DBIRecordset.fetchDict
 @see DBIRecordset.fetchObject
 */

FALCON_FUNC DBIRecordset_next( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->next() );
}

/*#
 @method fetchArray DBIRecordset
 @brief Get the next record as an Array.
 @return populated array or nil on EOF.

 @see DBIRecordset.next
 @see DBIRecordset.fetchDict
 @see DBIRecordset.fetchObject
*/

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
   dbi_type *cTypes = DBIHandle_getTypes( dbr );
   CoreArray *ary = new CoreArray( vm, cCount );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) ) {
         ary->append( i );
      } else {
         // TODO: handle error
      }
   }

   free( cTypes );
   vm->retval( ary );
}

/*#
 @method fetchDict DBIRecordset
 @brief Get the next record as a Dictionary.
 @return populated dictionary or nil on EOF.

 @see DBIRecordset.next
 @see DBIRecordset.fetchArray
 @see DBIRecordset.fetchObject
 */
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
   dbi_type *cTypes = DBIHandle_getTypes( dbr );

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
   }

   free( cTypes );
   free( cNames );

   vm->retval( dict );
}

/*#
 @method fetchObject DBIRecordset
 @brief Get the next record as an Object.
 @param obj Object to populate with row data.
 @return populated object or nil on EOF

 Please refer to the documentation of @a DBIRecord for more information on using
 the DBIRecord class.

 @see DBIRecordset.next
 @see DBIRecordset.fetchArray
 @see DBIRecordset.fetchDict
 @see DBIRecord
 */

FALCON_FUNC DBIRecordset_fetchObject( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *oI = vm->param( 0 );
   if ( oI == 0 || ! oI->isObject() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "0" ) ) );
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
   dbi_type *cTypes = DBIHandle_getTypes( dbr );

   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( DBIRecordset_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         o->setProperty( cNames[cIdx], i );
      } else {
         // TODO: handle error
      }
   }

   free( cTypes );
   free( cNames );

   vm->retval( o );
}

/*#
 @method getRowCount DBIRecordset
 @brief Get the number of rows in the recordset.
 @return Integer
 */

FALCON_FUNC DBIRecordset_getRowCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getRowCount() );
}

/*#
 @method getColumnTypes DBIRecordset
 @brief Get the column types as an array.
 @return Integer array of column types
 */

FALCON_FUNC DBIRecordset_getColumnTypes( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int cCount = dbr->getColumnCount();
   CoreArray *ary = new CoreArray( vm, cCount );
   dbi_type *cTypes = DBIHandle_getTypes( dbr );

   for (int cIdx=0; cIdx < cCount; cIdx++ )
      ary->append( (int64) cTypes[cIdx] );

   vm->retval( ary );

   free( cTypes );
}

/*#
 @method getColumnNames DBIRecordset
 @brief Get the column names as an array.
 @return String array of column names
 */

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
   }

   free( cNames );

   vm->retval( ary );
}

/*#
 @method getColumnCount DBIRecordset
 @brief Return the number of columns in the recordset.
 @return Integer
 */

FALCON_FUNC DBIRecordset_getColumnCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   vm->retval( dbr->getColumnCount() );
}


static void internal_asString_or_BlobID( VMachine *vm, int mode )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   Item *stringBuf = vm->param( 1 );

   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ||
        ( stringBuf != 0 && ! stringBuf->isString() )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N,[S]") ) );
      return;
   }


   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   String *value = stringBuf == 0 ?
            new GarbageString( vm ) :
            stringBuf->asString();

   dbi_status retval;
   if( mode == 0 )
      retval = dbr->asString( cIdx, *value );
   else
      retval = dbr->asBlobID( cIdx, *value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil();        // TODO: handle the error
   else
      vm->retval( (GarbageString *) value ); // we know it's a garbage string
}

/*#
 @method asString DBIRecordset
 @brief Get a field value as a String.
 @param idx field index
 @optparam string A pre-allocated string buffer that will be filled with the content of the field.
 @return String representation of database field contents or nil if field content is nil.

 If the @b string parameter is not provided, the function will create a new string
 and then return it; if it is provided, then the given string buffer will be
 filled with the contents of the field and also returned on succesful completion.

 Reusing the same item and filling it with new fetched values
 can be more efficient by several orders of magnitude.
 */

FALCON_FUNC DBIRecordset_asString( VMachine *vm )
{
   internal_asString_or_BlobID( vm, 0 );
}

/*#
 @method asBoolean DBIRecordset
 @brief Get a field value as a Boolean.
 @param idx field index
 @return Boolean representation of database field contents or nil if field content is nil
 */

FALCON_FUNC DBIRecordset_asBoolean( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N" ) ) );
      return;
   }

   bool value;

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   dbi_status retval = dbr->asBoolean( cIdx, value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
      vm->retnil();        // TODO: handle the error
   else {
      Item i( value );
      vm->retval( i );
   }
}

/*#
 @method asInteger DBIRecordset
 @brief Get a field value as an Integer.
 @param idx field index
 @return Integer representation of database field contents or nil if field content is nil
 */

FALCON_FUNC DBIRecordset_asInteger( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") ) );
      return;
   }

   int32 value;

   int32 cIdx = (int32) columnIndexI->asInteger();
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

/*#
 @method asInteger64 DBIRecordset
 @brief Get a field value as an Integer64.
 @param idx field index
 @return Integer64 representation of database field contents or nil if field content is nil
 */

FALCON_FUNC DBIRecordset_asInteger64( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") ) );
      return;
   }

   int64 value;

   int32 cIdx = (int32) columnIndexI->asInteger();
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

/*#
 @method asNumeric DBIRecordset
 @brief Get a field value as a Numeric.
 @param idx Field index.
 @return Numeric representation of database field contents or nil if field content is nil
 */

FALCON_FUNC DBIRecordset_asNumeric( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ) {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") ) );
      return;
   }

   numeric value;

   int32 cIdx = (int32) columnIndexI->asInteger();
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


// This is a little trick. A function mimicinzg three different date based functions
static void internal_asDate_or_time( VMachine *vm, int mode )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   Item *columnIndexI = vm->param( 0 );
   Item *i_timestamp = vm->param( 1 );
   CoreObject *timestamp;

   if ( columnIndexI == 0 || ! columnIndexI->isInteger() ||
        ( i_timestamp != 0 &&
           ( ! i_timestamp->isObject() ||
             ! (timestamp = i_timestamp->asObject() )->derivedFrom( "TimeStamp" )
           )
        )
      )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N,[TimeStamp]" ) ) );
      return;
   }

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( DBIRecordset_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   // create the timestamp -- or use the provided buffer?

   TimeStamp *ts = i_timestamp == 0 ?  new TimeStamp :
                   (TimeStamp *) timestamp->getUserData();

   dbi_status retval;
   switch( mode )
   {
      case 1:
         retval = dbr->asDate( cIdx, *ts );
      break;

      case 2:
         retval = dbr->asTime( cIdx, *ts );
      break;

      case 3:
         retval = dbr->asDateTime( cIdx, *ts );
      break;
   }

   if ( retval == dbi_nil_value )
   {
      if( i_timestamp == 0 ) delete ts;
      vm->retnil();
   }
   else if ( retval != dbi_ok )
   {
      // TODO: handle the error
      if( i_timestamp == 0 ) delete ts;
      vm->retnil();
   }
   else {
      if( i_timestamp == 0 )
      {
         // create falcon instance of the timestamp
         Item *ts_class = vm->findWKI( "TimeStamp" );
         fassert( ts_class != 0 );
         timestamp = ts_class->asClass()->createInstance();
         timestamp->setUserData( ts );
      }
      // else, our timestamp is already in place.

      vm->retval( timestamp );
   }
}

/*#
 @method asDate DBIRecordset
 @brief Get a field value as a TimeStamp object with the date populated and the time
      zeroed.
 @param idx field index
 @optparam timestamp A TimeStamp object that will be filled with the data in the field.
 @return TimeStamp representation of database field contents or nil if field content is nil

 If the @b timestamp parameter is not provided, the function will create a new instance
 of the TimeStamp class and then return it; if it is provided, then that object will be
 filled and also returned on succesful completion.

 Reusing the same object and filling it with new fetched values
 can be more efficient by several orders of magnitude.
*/

FALCON_FUNC DBIRecordset_asDate( VMachine *vm )
{
   internal_asDate_or_time( vm, 1 ); // get the date
}

/*#
 @method asTime DBIRecordset
 @brief Get a field value as a TimeStamp object with time populated and date zeroed.
 @param idx Field index.
 @optparam timestamp A TimeStamp object that will be filled with the data in the field.
 @return TimeStamp representation of database field contents or nil if field content is nil

 If the @b timestamp parameter is not provided, the function will create a new instance
 of the TimeStamp class and then return it; if it is provided, then that object will be
 filled and also returned on succesful completion.

 Reusing the same object and filling it with new fetched values
 can be more efficient by several orders of magnitude.
*/

FALCON_FUNC DBIRecordset_asTime( VMachine *vm )
{
   internal_asDate_or_time( vm, 2 ); // get the time
}

/*#
 @method asDateTime DBIRecordset
 @brief Get a field value as a TimeStamp object.
 @param idx field index
 @optparam timestamp A TimeStamp object that will be filled with the data in the field.
 @return TimeStamp representation of database field contents or nil if field content is nil

  If the @b timestamp parameter is not provided, the function will create a new instance
 of the TimeStamp class and then return it; if it is provided, then that object will be
 filled and also returned on succesful completion.

 Reusing the same object and filling it with new fetched values
 can be more efficient by several orders of magnitude.

*/

FALCON_FUNC DBIRecordset_asDateTime( VMachine *vm )
{
   internal_asDate_or_time( vm, 3 ); // get the date and the time
}

/*#
 @method asBlobID DBIRecordset
 @brief Get a field value as a blob ID string.
 @param idx field index
 @optparam bID A pre-allocated string buffer that will be filled with blob ID.
 @return A string value that can be used to retreive the content of a blob field.

 The value returned by this function can be then feed into blob related methods
 in the @a DBITransaction class, as the @a DBITransaction.readBlob method.

 If the @b bID parameter is not provided, the function will create a new string
 and then return it; if it is provided, then the given string buffer will be
 filled with the blob ID data, and eventually returned on success.

 Reusing the same item and filling it with new fetched values
 can be more efficient by several orders of magnitude.
*/

FALCON_FUNC DBIRecordset_asBlobID( VMachine *vm )
{
   internal_asString_or_BlobID( vm, 1 ); // specify it's a blob id
}


/*#
 @method getLastError DBIRecordset
 @brief Get the last error that occurred in this recordset from the database server.
 @return String containing error message

 This error message is specific to the database server type currently in use.
*/

FALCON_FUNC DBIRecordset_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   String value;
   dbi_status retval = dbr->getLastError( value );
   if ( retval != dbi_ok ) {
      vm->raiseModError( new DBIError( ErrorParam( retval, __LINE__ )
            .desc( "Could not get last error message" ) ) );
      return;
   }

   GarbageString *gs = new GarbageString( vm );
   gs->bufferize( value );
   vm->retval( gs );
}

/*#
 @method close DBIRecordset
 @brief Close a recordset
 */

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

/*#
 @method insert DBIRecord
 @brief Insert a new object into the database
 */

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

   DBIRecord_execute( vm, dbh, sql );
}

/*#
 @method update DBIRecord
 @brief Update an existing object in the database.
 */

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

   sql = "UPDATE " + *tableName + " SET ";

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

      sql += columnNames[cIdx] + " = " + value;
   }

   Item *primaryKeyValueI = self->getProperty( *primaryKey );
   String value;
   if ( DBIHandle_itemToSqlValue( dbh, primaryKeyValueI, value ) == 0 ) {

      vm->raiseModError( new DBIError( ErrorParam( dbi_invalid_type, __LINE__ )
                        .desc( "Invalid type for primary key" )
                        .extra(*primaryKey) ) );
      return;
   }

   sql.append( " WHERE " );
   sql.append( *primaryKey );
   sql.append( " = " );
   sql.append( value );

   DBIRecord_execute( vm, dbh, sql );
}

/*#
 @method delete DBIRecord
 @brief Delete an object from the database.
 */

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
   String sql = "DELETE FROM " + *tableName + " WHERE " + *primaryKey + " = " + value;
   DBIRecord_execute( vm, dbh, sql );
}

//======================================================
// DBI Blob Stream
//======================================================

/*#
 @method getBlobID DBIBlobStream
 @brief Returns the blob ID associated with this stream.
 @return A string containing the blob ID in this stream.
*/
FALCON_FUNC DBIBlobStream_getBlobID( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIBlobStream *bs = static_cast<DBIBlobStream *>( self->getUserData() );
   vm->retval( bs->getBlobID() );
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

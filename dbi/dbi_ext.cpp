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
#include "dbi_mod.h"

#include <dbiservice.h>

/*#
   @beginmodule dbi
*/

namespace Falcon {
namespace Ext {

CoreObject *dbi_defaultHandle; // Temporary until I figure how to set static class vars


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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S" ) );
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

         throw new DBIError( ErrorParam( DBI_ERROR_BASE + status, __LINE__ )
                                          .desc( "Uknown error (**)" )
                                          .extra( connectErrorMessage ) );

      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = provider->makeInstance( vm, hand );
      vm->retval( instance );

      dbi_defaultHandle = instance;
   }

   // no matter what we return if we had an error.
}


/**********************************************************
   Base transactional class
 **********************************************************/

/*#
 @method query DBIBaseTrans
 @brief Execute a SQL query.
 @param String SQL query
 @return an instance of @a DBIRecordset

 If the performed query doesn't generate a recordset, it
 returns nil.

*/

FALCON_FUNC DBIBaseTrans_query( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   
   DBIHandle* dbh;
   DBITransaction* dbt;
   String sql;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }
   
   if ( dbh_realSqlExpand( vm, dbh, sql, 0 ) == 0 )
      return;

   DBIRecordset *recSet = dbh_query_base( dbt, sql );

   // recordset-less query?
   if ( recSet != 0 )
      dbh_return_recordset( vm, recSet );
}


/*#
 @method queryOne DBIBaseTrans
 @brief Perform the SQL query and return the first field of the first record.
 @param String SQL query
 @return The value of the first field of the first record
 @raise DBIError if the query doesn't return any result.

 @see DBIBaseTrans.queryOneArray
 @see DBIBaseTrans.queryOneDict
 @see DBIBaseTrans.queryOneObject
 */

FALCON_FUNC DBIBaseTrans_queryOne( VMachine *vm )
{
   DBIRecordset *recSet = dbh_baseQueryOne( vm );
   if ( recSet == NULL ) {
      vm->retnil();
      return;
   }
   dbi_type *cTypes = recordset_getTypes( recSet );

   Item i;
   int32 id;
   recSet->asInteger( 0, id );
   int result;
   if ( (result = dbr_getItem( vm, recSet, cTypes[0], 0, i )) )
      vm->retval( i );

   recSet->close();
   memFree( cTypes );

   if ( ! result )
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_no_results, __LINE__ )
         .desc( "No results from query")
         .extra( "queryOne" ) );
}

/*#
 @method queryOneArray DBIBaseTrans
 @brief Perform the SQL query and return only the first record as an array.
 @param String SQL query
 @return Array populated array on nil on no results found.

 @see DBIBaseTrans.queryOne
 @see DBIBaseTrans.queryOneDict
 @see DBIBaseTrans.queryOneObject
 */

FALCON_FUNC DBIBaseTrans_queryOneArray( VMachine *vm )
{
   DBIRecordset *recSet = dbh_baseQueryOne( vm );

   int cCount = recSet->getColumnCount();
   CoreArray *ary = new CoreArray( cCount );
   dbi_type *cTypes = recordset_getTypes( recSet );

   for ( int cIdx=0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( dbr_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 )
      {
         recSet->close();
         memFree( cTypes );
         throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_no_results, __LINE__ )
            .desc( "No results from query")
            .extra( "queryOne" ) );
      }
      ary->append( i );
   }

   vm->retval( ary );
   memFree( cTypes );
}

/*#
 @method queryOneDict DBIBaseTrans
 @brief Perform the SQL query and return only the first record as a Dictionary.
 @param String SQL query
 @return Dictionary populated with result or nil on no results found.

 @see DBIBaseTrans.queryOne
 @see DBIBaseTrans.queryOneArray
 @see DBIBaseTrans.queryOneObject
 */

FALCON_FUNC DBIBaseTrans_queryOneDict( VMachine *vm )
{
   DBIRecordset *recSet = dbh_baseQueryOne( vm );

   int cCount = recSet->getColumnCount();
   PageDict *dict = new PageDict( cCount );
   char **cNames = (char **) memAlloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = recordset_getTypes( recSet );

   recSet->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( dbr_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 )
      {
         recSet->close();
         memFree( cTypes );
         memFree( cNames );
         throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_no_results, __LINE__ )
            .desc("No results from query")
            .extra("queryOneDict") );
      }

      CoreString *gsName = new CoreString;
      gsName->bufferize( cNames[cIdx] );

      dict->put( gsName, i );
   }

   memFree( cTypes );
   memFree( cNames );

   vm->retval( dict );
}

/*#
 @method queryOneObject DBIBaseTrans
 @brief Perform the SQL query and return only the first record as an Object.
 @param Object object to populate
 @param String SQL query
 @return populated object on success, or nil on no results found.

 Refer to documentation on @a DBIRecord for more information on using the DBI object
 query system.

 @see DBIBaseTrans.queryOne
 @see DBIBaseTrans.queryOneArray
 @see DBIBaseTrans.queryOneDict
 */

FALCON_FUNC DBIBaseTrans_queryOneObject( VMachine *vm )
{
   Item *objI = vm->param( 0 );
   if ( objI == 0 || ! objI->isObject() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "O" ) );
      return;
   }

   CoreObject *obj = objI->asObject();
   DBIRecordset *recSet = dbh_baseQueryOne( vm, 1);

   int cCount = recSet->getColumnCount();
   //TODO: Bufferize this things.
   char **cNames = (char **) memAlloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = recordset_getTypes( recSet );

   recSet->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( dbr_getItem( vm, recSet, cTypes[cIdx], cIdx, i ) == 0 ) {
         String indexString;
         indexString.writeNumber( (int64) cIdx );

         memFree( cTypes );
         memFree( cNames );
         throw new DBIError( ErrorParam( DBI_ERROR_BASE, __LINE__ )
                                 .desc( "Could not retrieve column value" )
                                 .extra( indexString ) );
      }

      obj->setProperty( cNames[cIdx], i );
   }

   memFree( cTypes );
   memFree( cNames );

   vm->retval( obj );
}


/*#
   @method insert DBIBaseTrans
   @brief Performs a single insertion query given a dictionary.
   @param table The table name on which to insert a new record.
   @param data The record to be inserted, as a dictionary.
   @raise DBIError if the query generates an error in the SQL engine.
   @raise ParamError if the data wasn't adequate to be inserted.

   This method automathises the task of creating the "insert" sql query
   given some variable data to be inserted. It actually expands into a
   complete SQL "insert" query that is then sent to the engine.

   @note Strings are expanded into utf-8 values, so the engine must
         support it.

   @see DBIBaseTrans.update
   @see DBIBaseTrans.delete
*/

FALCON_FUNC DBIBaseTrans_insert( VMachine *vm )
{
   Item* i_table = vm->param(0);
   Item* i_data = vm->param(1);

   if ( i_table == 0 || ! i_table->isString()
       || i_data == 0 || ! i_data->isDict() )
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra( "O" ) );
   }


   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   DBIHandle* dbh;
   DBITransaction* dbt;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   String sql = "insert into " + *i_table->asString() + "(";
   String vals = ") values (";

   Iterator iter( &i_data->asDict()->items() );
   bool bDone = false;
   while( iter.hasCurrent() )
   {
      String temp;
      if( iter.getCurrentKey().isString()  )
      {
         if ( ! bDone )
         {
            bDone = true;
         }
         else {
            sql += ", ";
            vals += ", ";
         }

         sql += *iter.getCurrentKey().asString();
         if ( ! dbh_itemToSqlValue( dbh, &iter.getCurrent(), temp ) )
         {
            bDone = false;
            break;
         }
         vals += temp;
      }

      iter.next();
   }

   sql += vals + ");";
   if ( bDone )
   {
      DBIRecordset *rec = dbh_query_base( dbt, sql  );
      if ( rec != 0 )
      {
         dbh_return_recordset( vm, rec );
      }
   }
   else {
       throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .extra( "could not generate an insert query" ) );
   }
}

/*#
   @method update DBIBaseTrans
   @brief Performs a single update query given a dictionary.
   @param table The table name on which to insert a new record.
   @param data The record to be inserted, as a dictionary.
   @raise DBIError if the query doesn't return any result.
   @raise ParamError if the data wasn't adequate to update the table.

   This method automathises the task of creating the "update" sql query
   given some variable data to be inserted. It actually expands into a
   complete SQL "update" query that is then sent to the engine.

   To determine a primary key or set of keys to insolate one (or more)
   records to be updated, declare the keys as OOB strings (i.e. via
   the oob() function). They won't be used in the 'SET' clause,
   and will form a 'where' clause in which they are all required
   (joint with an 'and' value).

   This function will refuse to update in case there isn't at least
   a key field in the dictionary.

   @see DBIBaseTrans.update
   @see DBIBaseTrans.delete
*/

FALCON_FUNC DBIBaseTrans_update( VMachine *vm )
{
   Item* i_table = vm->param(0);
   Item* i_data = vm->param(1);

   if ( i_table == 0 || ! i_table->isString()
       || i_data == 0 || ! i_data->isDict() )
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra( "O" ) );
   }


   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   DBIHandle* dbh;
   DBITransaction* dbt;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   String sql = "update " + *i_table->asString() + " set ";
   String where = "where ";

   Iterator iter( &i_data->asDict()->items() );
   bool bDone = false;
   bool bWhereDone = false;

   while( iter.hasCurrent() )
   {
      String temp;
      if( iter.getCurrentKey().isString()  )
      {
         if ( iter.getCurrentKey().isOob() )
         {
            // a key value
            if ( ! bWhereDone )
            {
               bWhereDone = true;
            }
            else {
               where += " and ";
            }
            dbh_itemToSqlValue( dbh, &iter.getCurrent(), temp );
            where += *iter.getCurrentKey().asString() + "="+temp;
         }
         else
         {
            if ( ! bDone )
            {
               bDone = true;
            }
            else {
               sql += ", ";
            }

            dbh_itemToSqlValue( dbh, &iter.getCurrent(), temp );
            sql += *iter.getCurrentKey().asString() + "="+temp;
         }
      }

      iter.next();
   }

   /*throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .extra( sql + "\n" + where ) );*/

   if ( ! bWhereDone )
   {
      throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .extra( "no key field designed for update" ) );
   }

   if ( bDone )
   {
      DBIRecordset *rec = dbh_query_base( dbt, sql + "\n" + where + ";" );
      if ( rec != 0 )
      {
         dbh_return_recordset( vm, rec );
      }
   }
   else
   {
       throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .extra( "could not generate an update query" ) );
   }
}


/*#
   @method delete DBIBaseTrans
   @brief Performs a single delete query given a dictionary.
   @param table The table name on which to insert a new record.
   @param data The keys indicating the record to be deleted, as a dictionary.
   @raise DBIError if the query doesn't return any result.
   @raise ParamError if the data wasn't adequate to delete a record.

   This method automathises the task of creating the "update" sql query
   given some variable data to be inserted. It actually expands into a
   complete SQL "update" query that is then sent to the engine.

   To determine a primary key or set of keys to insolate one (or more)
   records to be updated, declare the keys as OOB strings (i.e. via
   the oob() function). They won't be used in the 'SET' clause,
   and will form a 'where' clause in which they are all required
   (joint with an 'and' value).

   This function will refuse to update in case there isn't at least
   a key field in the dictionary.

   @see DBIBaseTrans.update
   @see DBIBaseTrans.delete
*/

FALCON_FUNC DBIBaseTrans_delete( VMachine *vm )
{
   Item* i_table = vm->param(0);
   Item* i_data = vm->param(1);

   if ( i_table == 0 || ! i_table->isString()
       || i_data == 0 || ! i_data->isDict() )
   {
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra( "O" ) );
   }


   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   DBIHandle* dbh;
   DBITransaction* dbt;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   String sql = "delete from " + *i_table->asString() + " where ";

   Iterator iter( &i_data->asDict()->items() );
   bool bWhereDone = false;

   while( iter.hasCurrent() )
   {
      String temp;
      if( iter.getCurrentKey().isString()  )
      {
         if ( ! bWhereDone )
         {
            bWhereDone = true;
         }
         else {
            sql += " and ";
         }
         sql += *iter.getCurrentKey().asString();

         dbh_itemToSqlValue( dbh, &iter.getCurrent(), temp );
         sql += *iter.getCurrentKey().asString() + "="+temp;
      }

      iter.next();
   }

   if ( bWhereDone )
   {
      DBIRecordset *rec = dbh_query_base( dbt, sql );
      if ( rec != 0 )
      {
         dbh_return_recordset( vm, rec );
      }
   }
   else
   {
       throw new ParamError( ErrorParam( e_param_range, __LINE__ )
               .extra( "could not generate a delete query" ) );
   }
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
 @method close DBITransaction
 @brief Close the current transaction handle.
 */

FALCON_FUNC DBITransaction_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->getHandle()->closeTransaction(dbt);
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
      dbh->getDefaultTransaction()->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + dbi_error, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }

   Item *trclass = vm->findWKI( "%DBITransaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
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
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                           .extra( "S" ) );
         return;
      }
      String sequenceName = *sequenceNameI->asString();
      vm->retval( dbh->getLastInsertedId( sequenceName ) );
   }
}

/*#
 @method getLastError DBIBaseTrans
 @brief Get the last error string from the database server.
 @return String

 This string is database server dependent. It is provided to get detailed information
 as to the error.
 */

FALCON_FUNC DBIBaseTrans_getLastError( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   
   DBIHandle* dbh;
   DBITransaction* dbt;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }


   String value;
   dbi_status retval = dbt->getLastError( value );
   if ( retval != dbi_ok ) {
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( "Unknown error" )
                                      .extra( "Could not get last error message " ) );
      return;
   }

   CoreString *gs = new CoreString;
   gs->bufferize( value );

   vm->retval( gs );
}

/*#
   @method sqlExpand DBIHandle
   @brief Expands a sql parameter string and its parameters into a complete string.
   @param sql The string to be expanded.
   @param ... The data that must be expanded.
   @return String

   This is what is usually done  by engines not supporting binding and positional
   parameters; :1, :2, :N markers are expanded into properly translated and
   escaped values taken from the coresponding parameter.
*/

FALCON_FUNC DBIHandle_sqlExpand( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   switch ( dbh->getQueryExpansionCapability() )
   {
      case DBIHandle::q_colon_sign_expansion:
         // Great, we have nothing to do; this are OK as they are.
         return;

      case DBIHandle::q_dollar_sign_expansion:
         // TODO: Build array and ship off to query method
         return;

      case DBIHandle::q_question_mark_expansion:
         // TODO: Convert :1, :2 into ?, ? and ship off to query method
         return;

      // We will handle default below
   }

   String sql;
   if ( dbh_realSqlExpand( vm, dbh, sql, 0 ) )
      vm->retval( new CoreString( sql ) );
}

/**********************************************************
 * Transaction class
 **********************************************************/
/*#
   @method openBlob DBITransaction
   @brief Opens an existing blob entity.
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S") );
      return;
   }

   dbi_status retval;
   DBIBlobStream *stream = dbt->openBlob( *i_blobID->asString(), retval );
   if ( stream == 0 ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
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
   Item *i_data = vm->param( 0 );
   Item *i_options = vm->param( 1 );
   if (
      ( i_data != 0 && ! (i_data->isString() || i_data->isMemBuf() || i_data->isNil() )) ||
      ( i_options != 0 && ! i_options->isString() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("[S|M],[S]") );
      return;
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   // are we willng to send a binary file?
   bool bBinary = i_data != 0 && i_data->isMemBuf();
   dbi_status status;
   DBIBlobStream *dbstream = dbt->createBlob(
      status,
      i_options == 0 ? "" : *i_options->asString(),
      bBinary );

   if ( dbstream == 0 )
   {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + status, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }

   // if we have no data to write, we're done.
   // Return an instance of the desired class
   if ( i_data == 0 || i_data->isNil() )
   {
      Item *i_instcls = vm->findWKI( dbstream->getFalconClassName() );
      // We suppose it's correctly set by the driver
      fassert( i_instcls != 0 );
      CoreObject *streamInst = i_instcls->asClass()->createInstance();
      streamInst->setUserData( dbstream );
      vm->retval( streamInst );
   }
   else {
      // we just have to write all the data and then close the stream.
      // If we have a string, write text...
      if( i_data->isString() )
      {
         dbstream->writeString( *i_data->asString() );
      }
      else {
         //... else write the stream binary
         MemBuf *mb = i_data->asMemBuf();
         // write bytewise... sorry, endianity will be lost
         dbstream->write( mb->data(), mb->size() );
      }

      // if we had an error, report it; but don't return, we must close.
      if( ! dbstream->good() )
      {
         String errorMessage;
         dbt->getLastError( errorMessage );
         throw new DBIError(
               ErrorParam( DBI_ERROR_BASE + (int32)dbstream->lastError(), __LINE__ )
                  .desc( errorMessage ) );
      }

      // anyhow close and delete
      dbstream->close();
      delete dbstream;
   }
}

/*#
   @method readBlob DBITransaction
   @brief Reads a whole blob entity.
   @param blobId The ID of the blob entity to be read.
   @optparam data A string or a MemBuf to be read.
   @optparam maxlen Maximum length read from the blob.
   @return A string containing the whole blob data, or read length if reading a MemBuf.
   @raise DBIError on error or if the ID is invalid.

   This method reads a whole blob in memory. If the parameter @b data is not given, then a new
   string, long enough to store all the blob, will be created. If it's given and it's a string,
   then the string buffer will be used, and it will be eventually expanded if needed. If it's
   a MemBuf, then the method will read at maximum the size of the MemBuf in bytes.

   The maximum size of the input may be also limited by the @b maxlen parameter. If both
   @b maxlen and @b data (as a MemBuf) are provided, the maximum size actually read will be
   the minimum of @b maxlen, the @b data MemBuf size and the blob entity size.

   When a MemBuf is read, then the method returns the amout of data actually imported. If the
   size of the MemBuf is not enough to contain the whole blob, it will be read as far as
   possible.

   @note It is possible to give a maximum length to be read and create dynamically the needed
   space by setting @b data to nil and passing @b maxlen.
*/
FALCON_FUNC DBITransaction_readBlob( VMachine *vm )
{
   Item *i_blobID = vm->param( 0 );
   Item *i_data = vm->param( 1 );
   Item *i_maxlen = vm->param( 2 );
   if (
      ( i_blobID == 0 || ! i_blobID->isString() ) ||
      ( i_data != 0 && ! (i_data->isString() || i_data->isMemBuf() || i_data->isNil() ) ) ||
      ( i_maxlen != 0 && ! i_maxlen->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S,[S|M],[N]") );
      return;
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   // open the blob
   dbi_status retval;
   DBIBlobStream *stream = dbt->openBlob( *i_blobID->asString(), retval );
   if ( stream == 0 ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }

   int64 absmax = i_maxlen != 0 ? i_maxlen->forceInteger() : -1;
   // eventually reduce the maximum if using a membuf
   if ( i_data != 0 && i_data->isMemBuf() )
   {
      MemBuf *readBuf = i_data->asMemBuf();

      if ( readBuf->size() < absmax || absmax == 0 )
      {
         absmax = readBuf->size();
      }

      int64 readIn = (int64) stream->read( readBuf, (int32) absmax );
      vm->retval( readIn );
   }
   else
   {
      // we must return a string.
      String *str = i_data == 0 || i_data->isNil() ? new CoreString : i_data->asString();

      // now we have our string. If absmax is 0, we must read till all is read.
      String temp(1024);
      str->size( 0 ); // be sure we are not appending.

      while( stream->readString( temp, 1024 ) )
      {
         *str += temp;
      }

      // error on read?
      vm->retval( str );
   }

   // raise also if retval is done; there's no problem with that.
   if ( ! stream->good() )
   {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError(
            ErrorParam( DBI_ERROR_BASE + (int32)stream->lastError(), __LINE__ )
               .desc( errorMessage ) );
   }

   stream->close();
   delete stream;
}

/*#
   @method writeBlob DBITransaction
   @brief Overwrites an existing blob entity.
   @param blobId The ID of the blob entity to be overwritten.
   @param data A string or a MemBuf to be written.
   @optparam start Character (if data is string) or byte (if data is MemBuf) from which to
      start writing.
   @optparam length Maximum count of  Characters (if @b data is a string) or bytes (if @b data a is MemBuf)
         to be written.
   @raise DBIError on error or if the ID is invalid.

   This method overwrites a whole existing blob entity. The selection range allows to
   write a portion of the existing data obviating the need to extract a subpart of it.
*/
FALCON_FUNC DBITransaction_writeBlob( VMachine *vm )
{
   Item *i_blobID = vm->param( 0 );
   Item *i_data = vm->param( 1 );
   Item *i_startFrom = vm->param( 2 );
   Item *i_maxlen = vm->param( 3 );

   if (
      ( i_blobID == 0 || ! i_blobID->isString() ) ||
      ( i_data == 0 || ! (i_data->isString() || i_data->isMemBuf()) ) ||
      ( i_startFrom != 0 && ! i_startFrom->isOrdinal() ) ||
      ( i_maxlen != 0 && ! i_maxlen->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                  .extra("S,[S|M],[N]") );
      return;
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );

   int64 startFrom = i_startFrom == 0 ? 0 : i_startFrom->forceInteger();
   int64 maxlen = i_maxlen == 0 ? String::npos : i_maxlen->forceInteger();
   if ( i_startFrom < 0 || maxlen <= 0 )
   {
      // nothing to write?
      throw new AccessError( ErrorParam( e_charRange, __LINE__ ) );
      return;
   }

    // open the blob
   dbi_status retval;
   DBIBlobStream *stream = dbt->openBlob( *i_blobID->asString(), retval );
   if ( stream == 0 ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }

   if ( i_data->isString() )
   {
      stream->writeString( *i_data->asString(), (uint32)startFrom, (uint32)maxlen );
   }
   else {
      MemBuf *mb = i_data->asMemBuf();
      if ( mb->size() < startFrom + maxlen )
      {
         stream->close();
         delete stream;
         // nothing to write?
         throw new AccessError( ErrorParam( e_charRange, __LINE__ ) );
         return;
      }

      stream->write( mb->data() + startFrom, (int32)maxlen );
   }

   if ( ! stream->good() )
   {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
   }

   stream->close();
   delete stream;
}


/*#
 @method commit DBIBaseTrans
 @brief Commit the transaction to the database.
 @raise DBIError on failure
 
 This method is available also on the database handler (main transaction),
 even if it's transaction-oriented, because some engine can provide a
 commit/rollback feature while not providing a full parallel transaction support.
 
 This does not close the transaction. You can perform a commit at safe steps within
 the transaction if necessary.
 */

FALCON_FUNC DBIBaseTrans_commit( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   
   DBIHandle* dbh;
   DBITransaction* dbt;
   String sql;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   dbi_status retval = dbt->commit();
   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }
}

/*#
 @method rollback DBIBaseTrans
 @brief Rollback the transaction (undo) to last commit point.
 @raise DBIError on failure

 This does not close the transaction. You can rollback and try another operation
 within the same transaction as many times as you wish.
 */

FALCON_FUNC DBIBaseTrans_rollback( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIItemBase *dbiitem = static_cast<DBIItemBase *>( self->getUserData() );
   
   DBIHandle* dbh;
   DBITransaction* dbt;
   String sql;
   
   if ( dbiitem->isHandle() )
   {
      dbh = static_cast<DBIHandle*>(dbiitem);
      dbt = dbh->getDefaultTransaction();
   }
   else {
      dbt = static_cast<DBITransaction*>(dbiitem);
      dbh = dbt->getHandle();
   }

   dbi_status retval = dbt->rollback();
   if ( retval != dbi_ok ) {
      String errorMessage;
      dbt->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
                                      .desc( errorMessage ) );
      return;
   }
}



/******************************************************************************
 * Recordset class
 *****************************************************************************/

/*#
 @method next DBIRecordset
 @brief Advanced the record pointer to the next record.
 @return true if successful, 0 if not successful, usually meaning the EOF has been
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

   vm->regA().setBoolean( dbr->next() == dbi_ok );
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
         throw new DBIError( ErrorParam( DBI_ERROR_BASE + nextRetVal, __LINE__ )
                                         .desc( errorMessage ) );
         return ;
      }
   }

   int cCount = dbr->getColumnCount();
   dbi_type *cTypes = recordset_getTypes( dbr );
   CoreArray *ary = new CoreArray( cCount );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      Item i;
      if ( dbr_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) ) {
         ary->append( i );
      } else {
         // TODO: handle error
      }
   }

   memFree( cTypes );
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
      {
      String errorMessage;
      dbr->getLastError( errorMessage );
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + nextRetVal, __LINE__ )
                                    .desc( errorMessage ) );
      return;
      }
   }

   int cCount = dbr->getColumnCount();
   PageDict *dict = new PageDict(  cCount );
   char **cNames = (char **) memAlloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = recordset_getTypes( dbr );

   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ )
   {
      CoreString *gsName = new CoreString;
      gsName->bufferize( cNames[cIdx] );

      Item i;
      if ( dbr_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         dict->put( gsName, i );
      } else {
         // TODO: handle error
      }
   }

   memFree( cTypes );
   memFree( cNames );

   vm->retval( new CoreDict( dict ) );
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "0" ) );
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
   char **cNames = (char **) memAlloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );
   dbi_type *cTypes = recordset_getTypes( dbr );

   dbr->getColumnNames( cNames );

   for ( int cIdx = 0; cIdx < cCount; cIdx++ ) {
      Item i;
      if ( dbr_getItem( vm, dbr, cTypes[cIdx], cIdx, i ) )
      {
         o->setProperty( cNames[cIdx], i );
      } else {
         // TODO: handle error
      }
   }

   memFree( cTypes );
   memFree( cNames );

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
   CoreArray *ary = new CoreArray( cCount );
   dbi_type *cTypes = recordset_getTypes( dbr );

   for (int cIdx=0; cIdx < cCount; cIdx++ )
      ary->append( (int64) cTypes[cIdx] );

   vm->retval( ary );

   memFree( cTypes );
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
   CoreArray *ary = new CoreArray( cCount );
   char **cNames = (char **) memAlloc( sizeof( char ) * cCount * DBI_MAX_COLUMN_NAME_SIZE );

   dbr->getColumnNames( cNames );

   for ( int cIdx=0; cIdx < cCount; cIdx++ ) {
      CoreString *gs = new CoreString;
      gs->bufferize( cNames[cIdx] );

      ary->append( gs );
   }

   memFree( cNames );

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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N,[S]") );
      return;
   }


   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
      return; // function handles reporting error to vm

   String *value = stringBuf == 0 ? new CoreString : stringBuf->asString();

   dbi_status retval;
   if( mode == 0 )
      retval = dbr->asString( cIdx, *value );
   else
      retval = dbr->asBlobID( cIdx, *value );

   if ( retval == dbi_nil_value )
      vm->retnil();
   else if ( retval != dbi_ok )
   {
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
            .desc( "Error while reading the recordset" ) );
   }
   else
      vm->retval( value ); // we know it's a garbage string
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N" ) );
      return;
   }

   bool value;

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") );
      return;
   }

   int32 value;

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") );
      return;
   }

   int64 value;

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra("N") );
      return;
   }

   numeric value;

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
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
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "N,[TimeStamp]" ) );
      return;
   }

   int32 cIdx = (int32) columnIndexI->asInteger();
   if ( dbr_checkValidColumn( vm, dbr, cIdx ) == 0 )
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
      throw new DBIError( ErrorParam( DBI_ERROR_BASE + retval, __LINE__ )
            .desc( "Could not get last error message" ) );
      return;
   }

   CoreString *gs = new CoreString;
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
   Item tableNameI, primaryKeyI, dbhI;
   self->getProperty( "_tableName", tableNameI );
   self->getProperty( "_primaryKey", primaryKeyI );
   self->getProperty( "_dbh", dbhI );

   CoreObject *dbhO = dbhI.asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI.asString();
   String *primaryKey = primaryKeyI.asString();

   int propertyCount = self->generator()->properties().size();
   String *columnNames = new String[propertyCount];

   propertyCount = dbr_getPersistPropertyNames( vm, self, columnNames, propertyCount );

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
      Item dummy;
      self->getProperty( columnNames[cIdx], dummy );
      if ( dbh_itemToSqlValue( dbh, &dummy, value ) == 0 ) {
         String errorMessage = "Invalid type for ";
         errorMessage.append( columnNames[cIdx] );

         throw new DBIError(
            ErrorParam( DBI_ERROR_BASE +dbi_invalid_type, __LINE__ )
                                         .desc( errorMessage ) );
         return;
      }

      sql.append( value );
   }

   sql.append( " )" );

   dbr_execute( vm, dbh, sql );
}

/*#
 @method update DBIRecord
 @brief Update an existing object in the database.
 */

FALCON_FUNC DBIRecord_update( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item tableNameI, primaryKeyI, dbhI;
   self->getProperty( "_tableName", tableNameI );
   self->getProperty( "_primaryKey", primaryKeyI );
   self->getProperty( "_dbh", dbhI );

   CoreObject *dbhO = dbhI.asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI.asString();
   String *primaryKey = primaryKeyI.asString();

   int propertyCount = self->generator()->properties().size();
   String *columnNames = new String[propertyCount];

   propertyCount = dbr_getPersistPropertyNames( vm, self, columnNames, propertyCount );

   String sql;

   sql = "UPDATE " + *tableName + " SET ";

   for ( int cIdx=0; cIdx < propertyCount; cIdx++ ) {
      if ( cIdx > 0 )
         sql.append( ", " );

      Item dummy;
      self->getProperty( columnNames[cIdx], dummy );

      String value;
      if ( dbh_itemToSqlValue( dbh, &dummy, value ) == 0 ) {
         String errorMessage = "Invalid type for ";
         errorMessage.append( columnNames[cIdx] );

         throw new DBIError(
               ErrorParam( DBI_ERROR_BASE + dbi_invalid_type, __LINE__ )
                                         .desc( errorMessage ) );
         return;
      }

      sql += columnNames[cIdx] + " = " + value;
   }

   Item primaryKeyValueI;
   String value;
   if ( ! self->getProperty( *primaryKey, primaryKeyValueI )
         || dbh_itemToSqlValue( dbh, &primaryKeyValueI, value ) == 0 ) {

      throw new DBIError(
            ErrorParam( DBI_ERROR_BASE + dbi_invalid_type, __LINE__ )
                        .desc( "Invalid type for primary key" )
                        .extra(*primaryKey) );
      return;
   }

   sql.append( " WHERE " );
   sql.append( *primaryKey );
   sql.append( " = " );
   sql.append( value );

   dbr_execute( vm, dbh, sql );
}

/*#
 @method delete DBIRecord
 @brief Delete an object from the database.
 */

FALCON_FUNC DBIRecord_delete( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item tableNameI, primaryKeyI, dbhI;
   self->getProperty( "_tableName", tableNameI );
   self->getProperty( "_primaryKey", primaryKeyI );
   self->getProperty( "_dbh", dbhI );

   CoreObject *dbhO = dbhI.asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( dbhO->getUserData() );

   String *tableName = tableNameI.asString();
   String *primaryKey = primaryKeyI.asString();
   Item pkValueI;
   self->getProperty( *primaryKey, pkValueI );
   String value;

   dbh_itemToSqlValue( dbh, &pkValueI, value );
   String sql = "DELETE FROM " + *tableName + " WHERE " + *primaryKey + " = " + value;
   dbr_execute( vm, dbh, sql );
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
      einst->setUserData( new DBIError );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of dbi_ext.cpp */

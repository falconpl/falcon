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
#include "dbi_st.h"

#include <dbiservice.h>

/*#
   @beginmodule dbi
*/

namespace Falcon {
namespace Ext {


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

void DBIConnect( VMachine *vm )
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
   // if it's 0, the service has already raised an error in the vm and we have nothing to do.
   fassert( provider != 0 );

   DBIHandle *hand = provider->connect( connString, false );

   // great, we have the database handler open. Now we must create a falcon object to store it.
   CoreObject *instance = provider->makeInstance( vm, hand );
   vm->retval( instance );
}


/**********************************************************
   Base transactional class
 **********************************************************/

static void internal_tropen( VMachine* vm, DBITransaction* trans )
{
   Item *trclass = vm->findWKI( "%Transaction" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}


static void internal_query_call( VMachine* vm, bool isQuery )
{
   Item* i_sql = vm->param(0);

   if ( i_sql == 0 || ! i_sql->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S, ..." ) );
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   int64 ar;

   ItemArray params( vm->paramCount() - 1 );
   for( int32 i = 1; i < vm->paramCount(); i++)
   {
      params.append( *vm->param(i) );
   }

   if( isQuery )
   {
      DBIRecordset* res;
      res = dbt->query( *i_sql->asString(), ar, params );
      fassert( res != 0 ); // else, query must raise.

      Item* rset_item = vm->findWKI( "%Recordset" );
      fassert( rset_item != 0 );
      fassert( rset_item->isClass() );

      CoreObject* rset = rset_item->asClass()->createInstance();
      rset->setUserData( res );

      vm->retval( rset );
   }
   else
   {
      dbt->call( *i_sql->asString(), ar, params );
   }
}

/*#
   @method query Transaction
   @brief Execute a SQL query bound to return a recordset.
   @param sql The SQL query
   @optparam ... Parameters for the query
   @return an instance of @a Recordset
   @throw DBIError if the database engine reports an error.

*/

void Transaction_query( VMachine *vm )
{
   internal_query_call( vm, true );
}

/*#
   @method call Transaction
   @brief Execute a SQL statement ignoring eventual recordsets.
   @param sql The SQL query
   @optparam ... Parameters for the query
   @throw DBIError if the database engine reports an error.
*/

void Transaction_call( VMachine *vm )
{
   internal_query_call( vm, false );
}

/*#
   @method prepare Transaction
   @brief Prepares a repeated statement.
   @param sql The SQL query
   @throw DBIError if the database engine reports an error.
*/

void Transaction_prepare( VMachine *vm )
{
   Item* i_sql = vm->param(0);

   if ( i_sql == 0 || ! i_sql->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S, ..." ) );
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->prepare( *i_sql->asString() );
}


/*#
   @method execute Transaction
   @brief Executes a repeated statement.
   @optparam ... The data to be passed to the repeated statement.
   @throw DBIError if the database engine reports an error.
*/

void Transaction_execute( VMachine *vm )
{
   ItemArray params( vm->paramCount() );
   for( int32 i = 0; i < vm->paramCount(); i++)
   {
      params.append( *vm->param(i) );
   }

   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->execute( params );
}


/*#
 @method close Transaction
 @brief Close the current transaction.
 */

void Transaction_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->close();
}

/*#
   @method tropen Transaction
   @brief Start a new sub-transaction of this transaction.
   @throw DBIError if the database engine doesn't support sub-transactions.

   Some database drivers allow to open sub-transactions whose effects are limited to the
   parent transaction, until it is finally committed in the database.

   If the database doesn't support sub-transactions, an error will be raised.
*/

void Transaction_tropen( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   internal_tropen( vm, dbt->startTransaction() );
}

/*#
 @method getLastID Transaction
 @brief Get the ID of the last record inserted.
 @optparam name A sequence name that is known by the engine.
 @return Integer

 This is database dependent but so widely used, it is included in the DBI module. Some
 databases such as MySQL only support getting the last inserted ID globally in the
 database server while others like PostgreSQL allow you to get the last inserted ID of
 any table. Thus, it is suggested that you always supply the sequence id as which to
 query. DBI drivers such as MySQL are programmed to ignore the extra information and
 return simply the last ID inserted into the database.
 */

void Transaction_getLastID( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbh = static_cast<DBITransaction *>( self->getUserData() );

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
 @method commit Transaction
 @brief Commit the  to the database.
 @raise DBIError on failure
 
 This method is available also on the database handler (main ),
 even if it's -oriented, because some engine can provide a
 commit/rollback feature while not providing a full parallel  support.
 
 This does not close the . You can perform a commit at safe steps within
 the  if necessary.
 */

void Transaction_commit( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->commit();
}

/*#
 @method rollback Transaction
 @brief Rollback the  (undo) to last commit point.
 @raise DBIError on failure

 This does not close the . You can rollback and try another operation
 within the same  as many times as you wish.
 */

void Transaction_rollback( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBITransaction *dbt = static_cast<DBITransaction *>( self->getUserData() );
   dbt->rollback();
}



/**********************************************************
   Handler class
 **********************************************************/

/*#
 @method tropen Handle
 @brief Start a transaction.
 @return an instance of @a Transaction

 This method returns a new transaction.
*/

void Handle_tropen( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   internal_tropen( vm, dbh->startTransaction() );
}

/*#
 @method close DBIHandle
 @brief Close the database handle.
 */

void Handle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
}



/******************************************************************************
 * Recordset class
 *****************************************************************************/

/*#
   @method fetch Recordset
   @brief Fetches a record and advances to the next.
   @optparam item Where to store the fetched record.
   @optparam count Number of rows fetched when @b item is a Table.
   @throw DBIError if the database engine reports an error.
   @return The @b item passed as a paramter filled with fetched data or
      nil when the recordset is terminated.

   If @b item is not given, a newly created array is filled with the
   fetched data and then returned, otherwise, the given item is filled
   with the data from the fetch.

   The @b item may be:
   - An Array.
   - A Dictionary.
   - A Table.

*/

void Recordset_fetch( VMachine *vm )
{
   Item *i_data = vm->param( 0 );
   Item *i_count = vm->param( 1 );

   // prepare the array in case of need.
   if( i_data == 0 )
   {
      vm->addLocals(1);
      i_data = vm->local(0);
      *i_data = new CoreArray();
      // if i_data is zero, then i_count is also zero, so we don't have to worry
      // about the VM stack being moved.
   }

   if ( ! ( i_data->isArray() || i_data->isDict() || i_data->isOfClass("Table") )
         || (i_count != 0 && ! i_count->isOrdinal())
         )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .extra( "[A|D|Table],[N]" ) );
   }

   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   if( ! dbr->fetchRow() )
   {
      vm->retnil();
      return;
   }


   if( i_data->isArray() )
   {
      int count = dbr->getColumnCount();
      CoreArray* aret = i_data->asArray();
      aret->resize( count );
      for ( int i = 0; i < count; i++ )
      {
         dbr->getColumnValue( i, aret->items()[i] );
      }
      vm->retval( aret );
   }
   else if( i_data->isDict() )
   {
      int count = dbr->getColumnCount();
      CoreDict* dret = i_data->asDict();
      for ( int i = 0; i < count; i++ )
      {
         String fieldName;
         dbr->getColumnName( i, fieldName );
         Item* value = dret->find( Item(&fieldName) );
         if( value == 0 )
         {
            Item v;
            dbr->getColumnValue( i, v );
            CoreString* key = new CoreString(fieldName);
            key->bufferize();
            dret->put( key, v );
         }
         else
         {
            dbr->getColumnValue( i, *value );
         }
      }
      vm->retval( dret );
   }
   else
   {
      // todo the table
   }

}

/*#
    @method discard Recordset
    @brief Discards one or more records in the recordset.
    @param count The number of records to be skipped.
    @return true if successful, false if the recordset is over.

    This skips the next @count records.
*/

void Recordset_discard( VMachine *vm )
{
   Item *i_count = vm->param( 0 );
   if ( i_count == 0 || ! i_count->isOrdinal() ) {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "N" ) );
   }

   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   vm->regA().setBoolean( dbr->discard( i_count->forceInteger() ) );
}


/*#
 @method getColumnNames DBIRecordset
 @brief Gets the names of the recordset columns.
 @return an array of one or more strings, containing the
    names of the rows in the record sect.
*/

void Recordset_getColumnNames( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   int count = dbr->getColumnCount();
   CoreArray* ret = new CoreArray( count );

   for( int i = 0; i < count; ++i )
   {
      CoreString* str = new CoreString;
      dbr->getColumnName( i, *str );
      str->bufferize();
      ret->append( str );
   }

   vm->retval( ret );
}

/*#
 @method getRowCount DBIRecordset
 @brief Return the number of columns in the recordset.
 @return An integer number >= 0 if the number of the current row is known,
    -1 if the driver can't access this information.
 */

void Recordset_getCurrentRow( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   vm->retval( dbr->getRowCount() );
}

/*#
 @method getRowCount DBIRecordset
 @brief Return the number of rows in the recordset.
 @return  An integer number >= 0 if the number of the current row is known,
    -1 if the driver can't access this information.
 */

void Recordset_getRowCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   vm->retval( dbr->getRowCount() );
}

/*#
 @method getColumnCount DBIRecordset
 @brief Return the number of columns in the recordset.
 @return  An integer number > 0.
 */

void Recordset_getColumnCount( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   vm->retval( dbr->getColumnCount() );
}


/*#
 @method close DBIRecordset
 @brief Close a recordset
 */

void Recordset_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );
   dbr->close();
}

//======================================================
// DBI error
//======================================================

void DBIError_init( VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new DBIError );

   ::Falcon::core::Error_init( vm );
}


}
}

/* end of dbi_ext.cpp */

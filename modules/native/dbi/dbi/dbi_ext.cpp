/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_ext.cpp
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo
 * Begin: Sun, 23 May 2010 20:17:58 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <stdio.h>
#include <string.h>

#include <falcon/engine.h>
#include <falcon/error.h>
#include <falcon/coretable.h>

#include "dbi.h"
#include "dbi_ext.h"
#include "dbi_st.h"

#include <falcon/dbi_common.h>

/*#
   @beginmodule dbi
*/

namespace Falcon {
namespace Ext {


/******************************************************************************
 * Main DBIConnect
 *****************************************************************************/

/*#
   @function connect
   @brief Connect to a database server through a DBI driver.
   @param conn SQL connection string.
   @optparam queryops Default transaction options to be applied to
                 operations performed on the returned handler returned handle.
   @return an instance of @a Handle.
   @raise DBIError if the connection fails.

  This function acts as a front-end to dynamically determine the DBI driver
  that should be used to connect to a determined database.
  
  The @b conn connection string is in the format described in @a dbi_load.
  An optional parameter @b queryops can be given to change some default
  value in the connection.
  
  @see Handle.options
*/

void DBIConnect( VMachine *vm )
{
   Item *paramsI = vm->param(0);
   Item *i_tropts = vm->param(1);
   if (  paramsI == 0 || ! paramsI->isString()
         || ( i_tropts != 0 && ! i_tropts->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                         .extra( "S,[S]" ) );
   }

   String *params = paramsI->asString();
   String provName = *params;
   String connString = "";
   uint32 colonPos = params->find( ":" );

   if ( colonPos != csh::npos )
   {
      provName = params->subString( 0, colonPos );
      connString = params->subString( colonPos + 1 );
   }

   DBIHandle *hand = 0;
   try
   {
      DBIService *provider = theDBIService.loadDbProvider( vm, provName );
      // if it's 0, the service has already raised an error in the vm and we have nothing to do.
      fassert( provider != 0 );

      hand = provider->connect( connString );
      if( i_tropts != 0 )
      {
         hand->options( *i_tropts->asString() );
      }

      // great, we have the database handler open. Now we must create a falcon object to store it.
      CoreObject *instance = provider->makeInstance( vm, hand );
      vm->retval( instance );
   }
   catch( DBIError* error )
   {
      delete hand;
      throw error;
   }
}


/**********************************************************
   Statement class
 **********************************************************/


/*#
   @method execute Statement
   @brief Executes a repeated statement.
   @optparam ... The data to be passed to the repeated statement.
   @return Number of rows affected by the command.
   @raise DBIError if the database engine reports an error.
   
   This method executes a statement that has been prepared through
   the @a Handle.prepare method. If the prepared statement
   could return a recordset, it is discarded (immediately closed
   server-side before it can reach the script). To receive a recordset
   use @a Statement.query
   
   @see Handle.prepare
*/

void Statement_execute( VMachine *vm )
{
   ItemArray params( vm->paramCount() );
   for( int32 i = 0; i < vm->paramCount(); i++)
   {
      params.append( *vm->param(i) );
   }

   CoreObject *self = vm->self().asObject();
   DBIStatement *dbt = static_cast<DBIStatement *>( self->getUserData() );
   vm->retval( dbt->execute( params ) );
}


/*#
   @method reset Statement
   @brief Resets this statement
   @raise DBIError if the statement cannot be reset.

   Some Database engines allow to reset a statement and retry to issue (execute) it
   without re-creating it anew.

   If the database engine doesn't support this feature, a DBIError will be thrown.
*/

void Statement_reset( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIStatement *dbt = static_cast<DBIStatement *>( self->getUserData() );
   dbt->reset();
}

/*#
    @method close Statement
    @brief Close this statement.

    Statements are automatically closed when the statement object
    is garbage collected, but calling explicitly this helps to
    reclaim data as soon as it is not necessary anymore.
*/

void Statement_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIStatement *dbt = static_cast<DBIStatement *>( self->getUserData() );
   dbt->close();
}


/**********************************************************
   Handler class
 **********************************************************/

/*#
   @method options Handle
   @brief Sets the default options for SQL operations performed on this handle.
   @param options The string containing the transaction options.
   @raise DBIError if the options are invalid.

   This method sets the default options that are used to create new transactions
   or performing statements.

   The options are set using a string where the settings are specified as <setting>=<value>
   pairs.

   Common options to all drivers include the followings:

    - prefetch: number of records to be pre-fetched at each query. The value
             may be "all" to wholly fetch queries locally, "none" to prefetch
             none or an arbitrary number of rows to be read from the server.
             By default, it's "all".
    - autocommit: Performs a transaction commit after each sql command.
                  Can be "on" or "off"; it's "off" by default.
    - cursor: Number of records returned by a query that should trigger the creation of a
              server side cursor. Can be "none" to prevent creation of server
              side cursor (the default) "all" to always create a cursor or an arbitrary
              number to create a server side cursor only if the query returns at least
              the indicated number of rows.
    - strings: All the data in the resultset is returned as a string, without
              transformation into Falcon items. In case the data is queried
              to just perform a direct output without particular needs for string
              formatting, or if the needed data formatting is performed by the engine,
              this option may improve performance considerably.
              Also, some data types may be not easily
              represented by Falcon types, and having the native engine representation
              may be crucial. If the database engine doesn't offer this option natively,
              the driver may ignore it or emulate it.

    Different database drivers may specify more transaction options; refer to their documentation
    for further parameters.
*/

void Handle_options( VMachine *vm )
{
   Item* i_options = vm->param(0);

   if( i_options == 0 || ! i_options->isString() )
   {
      throw new ParamError(ErrorParam( e_inv_params, __LINE__ )
           .extra( "S") );
   }

   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   dbh->options( *i_options->asString() );
}

/*#
   @method begin Handle
   @brief Issues a "begin work", "start transaction" or other appropriate command.
   @raise DBIError in case of error in starting the transaction.

   This method helps creating code portable across different database engines.
   It just issues the correct command for the database engine to start a transaction.

   It is not mandatory to manage transactions through this method, and this method
   can be intermixed with direct calls to @a Handle.perform calling the database
   engine commands directly.

   If the database engine doesn't support transaction, the command is ignored.
*/
void Handle_begin( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->begin();
}

/*#
   @method commit Handle
   @brief Issues a "commit work" command.
   @raise DBIError in case of error in starting the transaction.

   This method helps creating code portable across different database engines.
   It just issues the correct command for the database engine to commit the
   current transaction.

   It is not mandatory to manage transactions through this method, and this method
   can be intermixed with direct calls to @a Handle.perform calling the database
   engine commands directly.

   If the database engine doesn't support transaction, the command is ignored.
*/
void Handle_commit( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->commit();
}

/*#
   @method rollback Handle
   @brief Issues a "rollback work" command.
   @raise DBIError in case of error in starting the transaction.

   This method helps creating code portable across different database engines.
   It just issues the correct command for the database engine to roll back the
   current transaction.

   It is not mandatory to manage transactions through this method, and this method
   can be intermixed with direct calls to @a Handle.perform calling the database
   engine commands directly.

   If the database engine doesn't support transaction, the command is ignored.
*/
void Handle_rollback( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->rollback();
}

/*#
   @method lselect Handle
   @brief Returns a "select" query configured to access a sub-recordset.
   @param sql The query (excluded the "select" command).
   @optparam begin The first row to be returned (0 based).
   @optparam count The number of rows to be returned.
   @return A fully configured sql command that can be fed into @a Handle.query

   This method should create a "select" query adding the commands and/or the
    parameters needed by the engine to limit the resultset to a specified part
    part of the dataset.

    The query parameter must be a complete query EXCEPT for the "select" command,
    which is added by the engine. It must NOT terminate with a ";", which, in case
    of need is added by the engine.

    For example, to limit the following query to rows 5-14 (row 5 is the 6th row):
    @code
       SELECT field1, field2 FROM mytable WHERE key = ?;
    @endcode

    write this Falcon code:
    @code
       // dbh is a DBI handle
       rset = dbh.query(
                   dbh.lselect( "field1, field2 FROM mytable WHERE key = ?", 5, 10 ),
                   "Key value" )
    @endcode

    The @b count parameter can be 0 or @b nil to indicate "from @b begin to the end".
    It's not possible to return the n-last elements; to do that, reverse the
    query ordering logic.

    @note If the engine doesn't support limited recordsets, the limit parameters are
    ignored.
*/
void Handle_lselect( VMachine *vm )
{
   Item* i_sql = vm->param(0);
   Item* i_nBegin = vm->param(1);
   Item* i_nCount = vm->param(2);

   if( i_sql == 0 || ! i_sql->isString()
      || ( i_nBegin != 0 && ! (i_nBegin->isOrdinal() || i_nBegin->isNil() ) )
      || ( i_nCount != 0 && ! i_nCount->isOrdinal() )
         )
   {
      throw new ParamError(ErrorParam( e_inv_params, __LINE__ )
           .extra( "S,[N],[N]") );
   }

   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );

   CoreString* result = new CoreString;
   dbh->selectLimited( *i_sql->asString(),
         i_nBegin == 0 ? 0 : i_nBegin->forceInteger(),
         i_nCount == 0 ? 0 : i_nCount->forceInteger(), *result );

   vm->retval( result );
}

/*#
 @method close DBIHandle
 @brief Close the database handle.
 
  Every further operation on this object or on any related object will
  cause an exception to be raised.
*/

void Handle_close( VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   DBIHandle *dbh = static_cast<DBIHandle *>( self->getUserData() );
   dbh->close();
}

/*#
   @method getLastID Handle
   @brief Get the ID of the last record inserted.
   @optparam name A sequence name that is known by the engine.
   @return The value of the last single-field numeric key inserted in this transaction.

   This is database dependent but so widely used, it is included in the DBI module. Some
   databases such as MySQL only support getting the last inserted ID globally in the
   database server while others like PostgreSQL allow you to get the last inserted ID of
   any table. Thus, it is suggested that you always supply the sequence id as which to
   query. DBI drivers such as MySQL are programmed to ignore the extra information and
   return simply the last ID inserted into the database.
*/

void Handle_getLastID( VMachine *vm )
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


static void internal_stmt_open( VMachine* vm, DBIStatement* trans )
{
   Item *trclass = vm->findWKI( "%Statement" );
   fassert( trclass != 0 && trclass->isClass() );

   CoreObject *oth = trclass->asClass()->createInstance();
   oth->setUserData( trans );
   vm->retval( oth );
}


static void internal_query_call( VMachine* vm, int mode )
{
   Item* i_sql = vm->param(0);

   if ( i_sql == 0 || ! i_sql->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S, ..." ) );
   }

   CoreObject *self = vm->self().asObject();
   DBIHandle *dbt = static_cast<DBIHandle *>( self->getUserData() );

   ItemArray params( vm->paramCount() - 1 );
   for( int32 i = 1; i < vm->paramCount(); i++)
   {
      params.append( *vm->param(i) );
   }

   DBIRecordset* res = 0;

   int64 ar = -1;
   switch (mode)
   {
   case 0:  // query
      res = dbt->query( *i_sql->asString(), ar, params );
      break;

   case 1: // perform
      dbt->perform( *i_sql->asString(), ar, params );
      vm->retval( ar );
      break;

   case 2:  // call;
      res = dbt->call( *i_sql->asString(), ar, params );
      break;
   }

   self->setProperty("affected", ar );

   if( res !=0 )
   {
      Item* rset_item = vm->findWKI( "%Recordset" );
      fassert( rset_item != 0 );
      fassert( rset_item->isClass() );

      CoreObject* rset = rset_item->asClass()->createInstance();
      rset->setUserData( res );

      vm->retval( rset );
   }
}

/*#
   @method query Handle
   @brief Execute a SQL query bound to return a recordset.
   @param sql The SQL query
   @optparam ... Parameters for the query
   @return an instance of @a Recordset
   @raise DBIError if the database engine reports an error.

*/

void Handle_query( VMachine *vm )
{
   internal_query_call( vm, 0 );
}

/*#
   @method perform Handle
   @brief Execute a SQL statement ignoring eventual recordsets.
   @param sql The SQL query
   @optparam ... Parameters for the query
   @return Number of affected rows, or -1 if the data is not available.
   @raise DBIError if the database engine reports an error.

   Call this instead of query() when willing to perform SQL statements
   that are not supposed to return a recordset, or whose recordset must
   be ignored even if returned.
*/

void Handle_perform( VMachine *vm )
{
   internal_query_call( vm, 1 );
}

/*#
   @method prepare Handle
   @brief Prepares a repeated statement.
   @param sql The SQL query
   @raise DBIError if the database engine reports an error.

  This method creates a "prepared statement" that can be iteratively
  called with different parameters to perform multiple time the same
  operations. 
  
  Typically, the SQL statement will be a non-query data statement meant
  
*/

void Handle_prepare( VMachine *vm )
{
   Item* i_sql = vm->param(0);

   if ( i_sql == 0 || ! i_sql->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                        .extra( "S, ..." ) );
   }

   CoreObject *self = vm->self().asObject();
   DBIHandle *dbt = static_cast<DBIHandle *>( self->getUserData() );
   DBIStatement* stmt = dbt->prepare( *i_sql->asString() );
   internal_stmt_open(vm, stmt);
}


/*#
   @method call Handle
   @brief Calls a SQL stored procedure.
   @param sql The SQL query
   @optparam ... Parameters for the stored procedure.
   @raise DBIError if the database engine reports an error.

   Some engines have a special syntax for calling stored procedures which
   may return a recordset.

   This method asks the underlying driver to call the required stored procedure.
   If the SP generates a recordset, a @a Recordset object is returned, otherwise
   the method returns nil.
*/

void Handle_call( VMachine *vm )
{
   internal_query_call( vm, 2 );
}
/******************************************************************************
 * Recordset class
 *****************************************************************************/

static void internal_record_fetch( VMachine* vm, DBIRecordset* dbr, Item& target )
{
   int count = dbr->getColumnCount();

   if( target.isArray() )
   {
      CoreArray* aret = target.asArray();
      aret->resize( count );
      for ( int i = 0; i < count; i++ )
      {
         dbr->getColumnValue( i, aret->items()[i] );
      }
      vm->retval( aret );
   }
   else if( target.isDict() )
   {
      CoreDict* dret = target.asDict();
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
   /*
   else
   {
      CoreTable* tbl = dyncast<CoreTable*>(target.asObject()->getFalconData());
      ItemArray iaCols( count );

      if( tbl->order() == CoreTable::noitem )
      {
         String* fieldName = new String[count];
         for( int i = 0; i < count; ++ i )
         {
            dbr->getColumnName( i, fieldName[i] );
            iaCols.append( fieldName );
         }

         if( ! tbl->setHeader( iaCols ) )
         {
            delete[] fieldName;
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_FETCH, __LINE__ )
                  .extra("Incompatible table columns" ) );
         }

         delete[] fieldName;
      }
      else
      {
         if( tbl->order() != (unsigned) count )
         {
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_FETCH, __LINE__ )
                              .extra("Incompatible table columns" ) );
         }
      }

      // put in the values
      do {
         CoreArray* row = new CoreArray();
         row->resize( count );

         for( int i = 0; i < count; ++ i )
         {
            dbr->getColumnValue( i, row->at( i ) );
         }
         tbl->insertRow( row );
      }
      while( dbr->fetchRow() );

      vm->retval( target );
   }
   */
}

/*#
   @method fetch Recordset
   @brief Fetches a record and advances to the next.
   @optparam item Where to store the fetched record.
   @optparam count Number of rows fetched when @b item is a Table.
   @raise DBIError if the database engine reports an error.
   @return The @b item passed as a paramter filled with fetched data or
      nil when the recordset is terminated.

   If @b item is not given, a newly created array is filled with the
   fetched data and then returned, otherwise, the given item is filled
   with the data from the fetch.

   The @b item may be:
   - An Array.
   - A Dictionary.
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
/*
   if ( ! ( i_data->isArray() || i_data->isDict() || i_data->isOfClass("Table") )
         || (i_count != 0 && ! i_count->isOrdinal())
         )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .extra( "[A|D|Table],[N]" ) );
   }
*/

   if ( ! ( i_data->isArray() || i_data->isDict() )
         || (i_count != 0 && ! i_count->isOrdinal())
         )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
           .extra( "[A|D],[N]" ) );
   }

   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   if( ! dbr->fetchRow() )
   {
      vm->retnil();
      return;
   }

   internal_record_fetch( vm, dbr, *i_data );
}


/*#
    @method discard Recordset
    @brief Discards one or more records in the recordset.
    @param count The number of records to be skipped.
    @return true if successful, false if the recordset is over.

    This skips the next @b count records.
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
   @method getCurrentRow DBIRecordset
   @brief Returns the number of the current row.
   @return An integer number >= 0 if the number of the current row is known,
    -1 if the driver can't access this information.

    This method returns how many rows have been fetched before the current
    one. It will be -1 if the method @a Recordset.fetch has still not been called,
    0 if the current row is the first one, 1 for the second and so on.

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
   @return  An integer number >= 0 if the number of the row count row is known,
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

static bool Recordset_do_next( VMachine* vm )
{
   if( vm->regA().isOob() && vm->regA().isInteger() && vm->regA().asInteger() == 0 )
   {
      return false;
   }

   CoreObject *self = vm->self().asObject();
   DBIRecordset *dbr = static_cast<DBIRecordset *>( self->getUserData() );

   if( ! dbr->fetchRow() )
   {
      return false;
   }

   // copy, as we may disrupt the stack
   Item i_callable = *vm->param(0);

   if( vm->paramCount() == 1 )
   {
      int count = dbr->getColumnCount();
      for ( int i = 0; i < count; i++ )
      {
         Item value;
         dbr->getColumnValue( i, value );
         vm->pushParam( value );
      }

      vm->callFrame( i_callable, count );
   }
   else
   {
      internal_record_fetch( vm, dbr, *vm->param(1) );
      vm->pushParam( vm->regA() );
      vm->regA().setNil();
      vm->callFrame( i_callable, 1 );
   }

   return true;
}

/*#
   @method do Recordset
   @brief Calls back a function for each row in the recordset.
   @param cb The callback function that must be called for each row.
   @optparam item A fetchable item that will be filled and then passed to @b cb.
   @raise DBIError if the database engine reports an error.

   This method calls back a given @b cb callable item fetching one row at a time
   from the recordset, and then passing the data to @b cb either as parameters or
   as a single item.

   If @b item is not given, all the field values in the recordset are passed
   directly as parameters of the given @b cb function. If it is given, then
   that @b item is filled along the rules of @b Recordset.fetch and then
   it is passed to the @b cb item.

   The @b item may be:
   - An Array.
   - A Dictionary.
   - A Table.

   The @b cb method may return an oob(0) value to interrupt the processing of the
   recordset.

   @b The recordset is not rewinded before starting to call @b cb. Any previously
   fetched data won't be passed to @b cb.
*/

void Recordset_do( VMachine *vm )
{
   Item* i_callable = vm->param(0);
   Item* i_extra = vm->param(1);
   if( i_callable == 0 || ! i_callable->isCallable()
       || ( i_extra != 0
            && ! ( i_extra->isArray() || i_extra->isDict() || i_extra->isOfClass("Table") )
            )
     )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                              .extra( "C,[A|D|Table]" ) );
   }

   vm->regA().setNil();
   vm->returnHandler( Recordset_do_next );
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

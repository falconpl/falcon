/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_classhandle.cpp
 *
 * DBI Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Tue, 21 Jan 2014 16:38:11 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/dbi/dbi_classhandle.cpp"

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/error.h>
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>

#include "dbi.h"
#include "dbi_mod.h"

/*# @beginmodule dbi */

namespace Falcon {

namespace {
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

   FALCON_DECLARE_FUNCTION(options, "options:S")
   FALCON_DEFINE_FUNCTION_P1(options)
   {
      Item* i_options = ctx->param(0);

      if( i_options == 0 || ! i_options->isString() )
      {
         throw new ParamError(ErrorParam( e_inv_params, __LINE__ )
              .extra( "S") );
      }

      DBIHandle *dbh = ctx->tself<DBIHandle *>();

      dbh->options( *i_options->asString() );
      ctx->returnFrame();
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
   FALCON_DECLARE_FUNCTION(begin, "")
   FALCON_DEFINE_FUNCTION_P1(begin)
   {
      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      dbh->begin();
      ctx->returnFrame();
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
   FALCON_DECLARE_FUNCTION(commit, "")
   FALCON_DEFINE_FUNCTION_P1(commit)
   {
      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      dbh->commit();
      ctx->returnFrame();
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
   FALCON_DECLARE_FUNCTION(rollback, "")
   FALCON_DEFINE_FUNCTION_P1(rollback)
   {
      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      dbh->rollback();
      ctx->returnFrame();
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
   FALCON_DECLARE_FUNCTION(lselect, "sql:S,begin:[N],count:[N]")
   FALCON_DEFINE_FUNCTION_P1(lselect)
   {
      Item* i_sql = ctx->param(0);
      Item* i_nBegin = ctx->param(1);
      Item* i_nCount = ctx->param(2);

      if( i_sql == 0 || ! i_sql->isString()
         || ( i_nBegin != 0 && ! (i_nBegin->isOrdinal() || i_nBegin->isNil() ) )
         || ( i_nCount != 0 && ! i_nCount->isOrdinal() )
            )
      {
         throw paramError(__LINE__);
      }

      DBIHandle *dbh = ctx->tself<DBIHandle *>();

      String* result = new String;
      dbh->selectLimited( *i_sql->asString(),
            i_nBegin == 0 ? 0 : i_nBegin->forceInteger(),
            i_nCount == 0 ? 0 : i_nCount->forceInteger(), *result );

      ctx->returnFrame( FALCON_GC_HANDLE(result) );
   }

   /*#
    @method close DBIHandle
    @brief Close the database handle.

     Every further operation on this object or on any related object will
     cause an exception to be raised.
   */

   FALCON_DECLARE_FUNCTION(close, "")
   FALCON_DEFINE_FUNCTION_P1(close)
   {
      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      dbh->close();
      ctx->returnFrame();
   }

   /*# @property affected Handle

      Indicates the amount of rows affected by the last query performed on this
      connection.

      Will be 0 if none, -1 if unknown, or a positive value if the number of
      rows can be determined.
   */
   void get_affected(const Class*, const String&, void *instance, Item &property )
   {
      DBIHandle *dbt = static_cast<DBIHandle *>( instance );
      property = dbt->affectedRows();
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

   FALCON_DECLARE_FUNCTION(getLastID, "name:[S]")
   FALCON_DEFINE_FUNCTION_P1(getLastID)
   {
      DBIHandle *dbh = ctx->tself<DBIHandle *>();

      if ( ctx->paramCount() == 0 )
      {
         ctx->returnFrame( dbh->getLastInsertedId() );
      }
      else
      {
         Item *sequenceNameI = ctx->param( 0 );
         if ( sequenceNameI == 0 || ! sequenceNameI->isString() ) {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                              .extra( "S" ) );
            return;
         }

         String sequenceName = *sequenceNameI->asString();
         int64 lid = dbh->getLastInsertedId( sequenceName );
         ctx->returnFrame( lid );
      }
   }

   /*#
      @method query Handle
      @brief Execute a SQL query bound to return a recordset.
      @param sql The SQL query
      @optparam ... Parameters for the query
      @return an instance of @a Recordset, or nil.
      @raise DBIError if the database engine reports an error.

      On a succesful query, the property @a Handle.affected is
      assumes the count of affected rows, or -1 if the driver can't
      provide this information.

   */

   FALCON_DECLARE_FUNCTION(query, "...")
   FALCON_DEFINE_FUNCTION_P(query)
   {
      Item* i_sql = ctx->param(0);

      if ( i_sql == 0 || ! i_sql->isString() )
      {
         throw paramError(__LINE__);
      }

      DBIHandle *dbt = ctx->tself<DBIHandle *>();

      DBIRecordset* res = 0;
      if( pCount > 1 )
      {
         ItemArray params( pCount - 1 );
         for( int32 i = 1; i < pCount; i++)
         {
            params.append( *ctx->param(i) );
         }

         // Query may throw.
         res = dbt->query( *i_sql->asString(), &params );
      }
      else
      {
         res = dbt->query( *i_sql->asString() );
      }

      if( res != 0 )
      {
         DBIModule* dbm = static_cast<DBIModule*>(this->methodOf()->module());
         Class* cls = dbm->recordsetClass();
         ctx->returnFrame( FALCON_GC_STORE(cls, res) );
      }
      else {
         ctx->returnFrame();
      }
   }


   /*#
      @method aquery Handle
      @brief Execute a SQL query bound to return a recordset.
      @param sql The SQL query
      @param params Values to be passed to the query in an array.
      @return an instance of @a Recordset, or nil.
      @raise DBIError if the database engine reports an error.

      On a succesful query, the property @a Handle.affected is
      assumes the count of affected rows, or -1 if the driver can't
      provide this information.
   */

   FALCON_DECLARE_FUNCTION(aquery, "params:[A]")
   FALCON_DEFINE_FUNCTION_P1(aquery)
   {
      Item* i_sql = ctx->param(0);
      Item* i_params = ctx->param(1);
      if ( i_sql == 0 || ! i_sql->isString()
         || i_params == 0 || ! i_params->isArray() )
      {
         throw paramError(__LINE__);
      }

      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      DBIRecordset* res = dbh->query( *i_sql->asString(), i_params->asArray() );

      if( res != 0 )
      {
         DBIModule* dbm = static_cast<DBIModule*>(this->methodOf()->module());
         Class* cls = dbm->recordsetClass();
         ctx->returnFrame( FALCON_GC_STORE(cls, res) );
      }
      else {
         ctx->returnFrame();
      }
   }

   /*#
      @method expand Handle
      @brief Expands a sql query with provided parameters.
      @param sql The SQL query
      @optparam ... Parameters for the query
      @return A string containing a complete SQL statement.
      @raise DBIError in case the expansion fails.

      Some underlying database engine may not be consistently working
      with the Falcon types used as parameters. As such, this utility
      can be used to create complete query strings that doesn't require
      driver-side parameter binding and expansion.

   */

   FALCON_DECLARE_FUNCTION(expand, "sql:S,...")
   FALCON_DEFINE_FUNCTION_P(expand)
   {
      Item* i_sql = ctx->param(0);

      if ( i_sql == 0 || ! i_sql->isString() )
      {
         throw paramError(__LINE__);
      }

      DBIHandle *dbh = ctx->tself<DBIHandle *>();
      String* target = new String;

      ItemArray params( pCount - 1 );
      for( int32 i = 1; i < pCount; i++)
      {
         params.append( *ctx->param(i) );
      }

      Item ret = FALCON_GC_HANDLE(target);
      // May throw
      dbh->sqlExpand( *i_sql->asString(), *target, params );
      ctx->returnFrame( ret );
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
   FALCON_DECLARE_FUNCTION(prepare, "sql:S")
   FALCON_DEFINE_FUNCTION_P1(prepare)
   {
      Item* i_sql = ctx->param(0);

      if ( i_sql == 0 || ! i_sql->isString() )
      {
         throw paramError(__LINE__);
      }

      DBIHandle *dbt = ctx->tself<DBIHandle *>();
      DBIStatement* stmt = dbt->prepare( *i_sql->asString() );
      DBIModule* dbm = static_cast<DBIModule*>(methodOf()->module());
      Class* cls = dbm->statementClass();
      ctx->returnFrame( FALCON_GC_STORE(cls, stmt) );
   }
}

//===========================================================================
//
//===========================================================================

/*#
   @class Handle
   @brief DBI connection handle returned by @a connect.

   This is the main database interface connection abstraction,
   which allows to issue SQL statements and inspect
   the result of SQL queries.
 */
ClassHandle::ClassHandle():
      Class(FALCON_DBI_HANDLE_CLASS_NAME)
{
   addMethod( new FALCON_FUNCTION_NAME(options) );
   addMethod( new FALCON_FUNCTION_NAME(begin) );
   addMethod( new FALCON_FUNCTION_NAME(commit) );
   addMethod( new FALCON_FUNCTION_NAME(rollback) );
   addMethod( new FALCON_FUNCTION_NAME(lselect) );
   addMethod( new FALCON_FUNCTION_NAME(close) );
   addMethod( new FALCON_FUNCTION_NAME(getLastID) );
   addMethod( new FALCON_FUNCTION_NAME(query) );
   addMethod( new FALCON_FUNCTION_NAME(aquery) );
   addMethod( new FALCON_FUNCTION_NAME(expand) );
   addMethod( new FALCON_FUNCTION_NAME(prepare) );

   addProperty( "affected", &get_affected );
}

ClassHandle::~ClassHandle()
{}

void ClassHandle::dispose( void* instance ) const
{
   DBIHandle* dbh = static_cast<DBIHandle*>(instance);
   delete dbh;
}

void* ClassHandle::clone( void* ) const
{
   return 0;
}

void* ClassHandle::createInstance() const
{
   return 0;
}

void ClassHandle::gcMarkInstance( void* instance, uint32 mark ) const
{
   DBIHandle* dbh = static_cast<DBIHandle*>(instance);
   dbh->gcMark(mark);
}

bool ClassHandle::gcCheckInstance( void* instance, uint32 mark ) const
{
   DBIHandle* dbh = static_cast<DBIHandle*>(instance);
   return dbh->currentMark() >= mark;
}

}

/* end of dbi_classhandle.cpp */

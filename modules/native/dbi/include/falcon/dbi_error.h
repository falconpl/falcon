/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_error.h

   Database Interface - Error management
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 23:47:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_DBI_ERROR_H_
#define FALCON_DBI_ERROR_H_

#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_DBI_ERROR_BASE
   #define FALCON_DBI_ERROR_BASE 2000
#endif

#define FALCON_DBI_ERROR_COLUMN_RANGE     (FALCON_DBI_ERROR_BASE+1)
#define FALCON_DBI_ERROR_INVALID_DRIVER   (FALCON_DBI_ERROR_BASE+2)
#define FALCON_DBI_ERROR_NOMEM            (FALCON_DBI_ERROR_BASE+3)
#define FALCON_DBI_ERROR_CONNPARAMS       (FALCON_DBI_ERROR_BASE+4)
#define FALCON_DBI_ERROR_CONNECT          (FALCON_DBI_ERROR_BASE+5)
#define FALCON_DBI_ERROR_QUERY            (FALCON_DBI_ERROR_BASE+6)
#define FALCON_DBI_ERROR_QUERY_EMPTY      (FALCON_DBI_ERROR_BASE+7)
#define FALCON_DBI_ERROR_OPTPARAMS        (FALCON_DBI_ERROR_BASE+8)
#define FALCON_DBI_ERROR_NO_SUBTRANS      (FALCON_DBI_ERROR_BASE+9)
#define FALCON_DBI_ERROR_NO_MULTITRANS    (FALCON_DBI_ERROR_BASE+10)
#define FALCON_DBI_ERROR_UNPREP_EXEC      (FALCON_DBI_ERROR_BASE+11)
#define FALCON_DBI_ERROR_BIND_SIZE        (FALCON_DBI_ERROR_BASE+12)
#define FALCON_DBI_ERROR_BIND_MIX         (FALCON_DBI_ERROR_BASE+13)
#define FALCON_DBI_ERROR_EXEC             (FALCON_DBI_ERROR_BASE+14)
#define FALCON_DBI_ERROR_FETCH            (FALCON_DBI_ERROR_BASE+15)
#define FALCON_DBI_ERROR_UNHANDLED_TYPE   (FALCON_DBI_ERROR_BASE+16)
#define FALCON_DBI_ERROR_RESET            (FALCON_DBI_ERROR_BASE+17)
#define FALCON_DBI_ERROR_BIND_INTERNAL    (FALCON_DBI_ERROR_BASE+18)
#define FALCON_DBI_ERROR_TRANSACTION      (FALCON_DBI_ERROR_BASE+19)
#define FALCON_DBI_ERROR_CLOSED_STMT      (FALCON_DBI_ERROR_BASE+20)
#define FALCON_DBI_ERROR_CLOSED_RSET      (FALCON_DBI_ERROR_BASE+21)
#define FALCON_DBI_ERROR_CLOSED_DB        (FALCON_DBI_ERROR_BASE+22)
#define FALCON_DBI_ERROR_DB_NOTFOUND      (FALCON_DBI_ERROR_BASE+23)
#define FALCON_DBI_ERROR_CONNECT_CREATE   (FALCON_DBI_ERROR_BASE+24)

namespace Falcon
{

/** Base error class for all DBI errors.

    DBI Error descriptions are available in English ONLY, until
    the new per-module string table support is ready.
 */


   /*#
    @class DBIError
    @brief DBI specific error.

    Inherited class from Error to distinguish from a standard Falcon error. In many
    cases, DBIError.extra will contain the SQL query that caused the problem.

    Error code is one of the following:
    - DBIError.COLUMN_RANGE: Column out of range
    - DBIError.INVALID_DRIVER: DBI driver service not found
    - DBIError.NOMEM: Not enough memory to perform the operation
    - DBIError.CONNPARAMS: Malformed or invalid connection parameter string
    - DBIError.CONNECT: Connection to database failed
    - DBIError.QUERY: Database query error
    - DBIError.QUERY_EMPTY: Query didn't return any result
    - DBIError.OPTPARAMS: Unrecognized or invalid options
    - DBIError.NO_SUBTRANS: DBEngine doesn't support sub-transactions
    - DBIError.NO_MULTITRANS: DBEngine doesn't support multiple transactions
    - DBIError.UNPREP_EXEC: Called 'execute' without having previously called 'prepare'
    - DBIError.BIND_SIZE: Input variables in 'execute' and statement parameters have different size
    - DBIError.BIND_MIX: Input variables passed in 'execute' cannot be bound to the statement
    - DBIError.EXEC: Error during an 'execute' on a prepared statement
    - DBIError.FETCH: Failed to fetch part of the recordset
    - DBIError.UNHANDLED_TYPE: Unhandled field type in return dataset
    - DBIError.RESET: Error while resetting a statement
    - DBIError.BIND_INTERNAL: Internal SQL expansion failed
    - DBIError.TRANSACTION: Error in issuing standard transactional command
    - DBIError.CLOSED_STMT: Statement already closed
    - DBIError.CLOSED_RSET: Recordset already closed
    - DBIError.CLOSED_DB: DB already closed
    - DBIError.DB_NOTFOUND: Requested database not found
    - DBIError.CONNECT_CREATE: Unable to create the database as required
   */

   FALCON_DECLARE_ERROR_INSTANCE_WITH_DESC( DBIError,
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_COLUMN_RANGE, "Column out of range" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_INVALID_DRIVER, "DBI driver service not found or failed to load"  )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_NOMEM, "Not enough memory to perform the operation" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CONNPARAMS, "Malformed or invalid connection parameter string" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CONNECT, "Connection to database failed" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_QUERY, "Database query error" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_QUERY_EMPTY, "Query didn't return any result" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_OPTPARAMS, "Unrecognized or invalid options" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_NO_SUBTRANS, "DBEngine doesn't support sub-transactions" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_NO_MULTITRANS, "DBEngine doesn't support multiple transactions" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_UNPREP_EXEC, "Called 'execute' without having previously called 'prepare'" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_BIND_SIZE, "Input variables in 'execute' and statement parameters have different size" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_BIND_MIX, "Input variables passed in 'execute' cannot be bound to the statement" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_EXEC, "Error during an 'execute' on a prepared statement" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_FETCH, "Failed to fetch part of the recordset" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_UNHANDLED_TYPE, "Unhandled field type in return dataset" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_RESET, "Error while resetting a statement" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_BIND_INTERNAL, "Internal SQL expansion failed" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_TRANSACTION, "Error in issuing standard transactional command" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CLOSED_STMT, "Statement already closed" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CLOSED_RSET, "Recordset already closed" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CLOSED_DB, "DB already closed" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_DB_NOTFOUND, "Requested database not found" )
            FALCON_ERROR_CLASS_DESC( FALCON_DBI_ERROR_CONNECT_CREATE, "Unable to create the database as required" )
   );

   FALCON_DECLARE_ERROR_CLASS_EX( DBIError, \
            addConstant("COLUMN_RANGE", FALCON_DBI_ERROR_COLUMN_RANGE );\
            addConstant("INVALID_DRIVER", FALCON_DBI_ERROR_INVALID_DRIVER );\
            addConstant("NOMEM", FALCON_DBI_ERROR_NOMEM );\
            addConstant("CONNPARAMS", FALCON_DBI_ERROR_CONNPARAMS );\
            addConstant("CONNECT", FALCON_DBI_ERROR_CONNECT );\
            addConstant("QUERY", FALCON_DBI_ERROR_QUERY );\
            addConstant("QUERY_EMPTY", FALCON_DBI_ERROR_QUERY_EMPTY );\
            addConstant("OPTPARAMS", FALCON_DBI_ERROR_OPTPARAMS );\
            addConstant("NO_SUBTRANS", FALCON_DBI_ERROR_NO_SUBTRANS );\
            addConstant("NO_MULTITRANS", FALCON_DBI_ERROR_NO_MULTITRANS );\
            addConstant("UNPREP_EXEC", FALCON_DBI_ERROR_UNPREP_EXEC );\
            addConstant("BIND_SIZE", FALCON_DBI_ERROR_BIND_SIZE );\
            addConstant("BIND_MIX", FALCON_DBI_ERROR_BIND_MIX );\
            addConstant("EXEC", FALCON_DBI_ERROR_EXEC );\
            addConstant("FETCH", FALCON_DBI_ERROR_FETCH );\
            addConstant("UNHANDLED_TYPE", FALCON_DBI_ERROR_UNHANDLED_TYPE );\
            addConstant("RESET", FALCON_DBI_ERROR_RESET );\
            addConstant("BIND_INTERNAL", FALCON_DBI_ERROR_BIND_INTERNAL );\
            addConstant("TRANSACTION", FALCON_DBI_ERROR_TRANSACTION );\
            addConstant("CLOSED_STMT", FALCON_DBI_ERROR_CLOSED_STMT );\
            addConstant("CLOSED_RSET", FALCON_DBI_ERROR_CLOSED_RSET );\
            addConstant("CLOSED_DB", FALCON_DBI_ERROR_CLOSED_DB );\
            addConstant("DB_NOTFOUND", FALCON_DBI_ERROR_DB_NOTFOUND );\
            addConstant("CONNECT_CREATE", FALCON_DBI_ERROR_CONNECT_CREATE );\
            )
}

#endif

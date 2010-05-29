/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_error.cpp

   Database Interface - Error management
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 23:47:36 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_error.h>

namespace Falcon {

DBIError::DBIError( const ErrorParam &params  ):
   Error( "DBIError", params )
{
   describeError();
}

void DBIError::describeError()
{
   switch( errorCode() )
   {
   case FALCON_DBI_ERROR_COLUMN_RANGE:
      this->errorDescription( "Column out of range" );
      break;

   case FALCON_DBI_ERROR_INVALID_DRIVER:
      this->errorDescription( "DBI driver service not found" );
      break;

   case FALCON_DBI_ERROR_NOMEM:
      this->errorDescription( "Not enough memory to perform the operation" );
      break;

   case FALCON_DBI_ERROR_CONNPARAMS:
      this->errorDescription( "Malformed or invalid connection parameter string" );
      break;

   case FALCON_DBI_ERROR_CONNECT:
      this->errorDescription( "Connection to database failed" );
      break;

   case FALCON_DBI_ERROR_QUERY:
      this->errorDescription( "Database query error" );
      break;

   case FALCON_DBI_ERROR_QUERY_EMPTY:
      this->errorDescription( "Query didn't return any result" );
      break;

   case FALCON_DBI_ERROR_OPTPARAMS:
      this->errorDescription( "Unrecognized or invalid options" );
      break;

   case FALCON_DBI_ERROR_NO_SUBTRANS:
      this->errorDescription( "DBEngine doesn't support sub-transactions" );
      break;

   case FALCON_DBI_ERROR_NO_MULTITRANS:
      this->errorDescription( "DBEngine doesn't support multiple transactions" );
      break;

   case FALCON_DBI_ERROR_UNPREP_EXEC:
      this->errorDescription( "Called 'execute' without having previously called 'prepare'" );
      break;

   case FALCON_DBI_ERROR_BIND_SIZE:
      this->errorDescription( "Input variables in 'execute' and statement parameters have different size" );
      break;

   case FALCON_DBI_ERROR_BIND_MIX:
      this->errorDescription( "Input variables passed in 'execute' cannot be bound to the statement" );
      break;

   case FALCON_DBI_ERROR_EXEC:
      this->errorDescription( "Error during an 'execute' on a prepared statement" );
      break;

   case FALCON_DBI_ERROR_FETCH:
      this->errorDescription( "Failed to fetch part of the recordset" );
      break;

   case FALCON_DBI_ERROR_UNHANDLED_TYPE:
      this->errorDescription( "Unhandled field type in return dataset" );
      break;

   case FALCON_DBI_ERROR_RESET:
      this->errorDescription( "Error while resetting a statement" );
      break;

   case FALCON_DBI_ERROR_BIND_INTERNAL:
      this->errorDescription( "Internal SQL expansion failed" );
      break;

   case FALCON_DBI_ERROR_TRANSACTION:
      this->errorDescription( "Error in issuing standard transactional command" );
      break;

      // by default, do nothing -- let the base system to put an appropriate description
   }
}

}

/* end of dbi_error.cpp */

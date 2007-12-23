/*
   FALCON - The Falcon Programming Language
   FILE: pgsql_mod.cpp

   PgSQL DBI module -- module service classes
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: 2007-12-22 18:22
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   pgsql_mod.cpp - PgSQL DBI module -- module service classes
*/

#include <falcon/string.h>
#include <falcon/autocstring.h>

#include "pgsql_mod.h"

namespace Falcon
{
   int PgSQLConnection::connect( const String *connString )
   {
      AutoCString asConnString( connString );
      m_conn = PQconnectdb( asConnString.c_str() );

      if (m_conn == NULL)
      {
         return setErrorInfo( DBI_MEMORY_ALLOC_ERROR,
                              "Memory allocation error" );
      }
      else if ( PQstatus( m_conn ) != CONNECTION_OK )
      {
         return setErrorInfo( DBI_CONNECTION_ERROR,
                              PQerrorMessage( m_conn ) );
      }

      return DBI_OK;
   }

   int PgSQLConnection::beginTransaction()
   {
      return execute( "BEGIN" );
   }

   int PgSQLConnection::rollbackTransaction()
   {
      return execute( "ROLLBACK" );
   }

   int PgSQLConnection::commitTransaction()
   {
      return execute( "COMMIT" );
   }

   int PgSQLConnection::execute( const String *sql )
   {
      if ( m_conn == NULL )
      {
         return setErrorInfo( DBI_NO_CONNECTION, "No database connection" );
      }

      AutoCString asSql( sql );
      PGresult *res =  PQexec( m_conn, asSql.c_str() );
      if ( res == NULL )
      {
         return setErrorInfo( DBI_MEMORY_ALLOC_ERROR,
                              "Memory allocation error" );
      }

      switch ( PQresultStatus( res ))
      {
      case PGRES_EMPTY_QUERY:
         setErrorInfo( DBI_OK, "" );
         m_affectedRows = 0;
         PQclear( res );

         return m_errorCode;

      case PGRES_COMMAND_OK:
         setErrorInfo( DBI_OK, "" );
         m_affectedRows = PQcmdTuples( res );
         PQclear( res );

         return m_errorCode;

      case PGRES_TUPLES_OK:
      case PGRES_COPY_OUT:
      case PGRES_COPY_IN:
         setErrorInfo( DBI_OK, "" );
         PQclear( res );

         return m_errorCode;

      default:
         setErrorInfo( DBI_QUERY_ERROR, PQresultErrorMessage( res ) );
         PQclear( res );

         return m_errorCode;
      }
   }

   PgSQLRecordset *PgSQLConnection::query( const String *sql )
   {
      if ( m_conn == NULL )
      {
         setErrorInfo( DBI_NO_CONNECTION, "No database connection" );
         return NULL;
      }

      AutoCString asSql( sql );
      PGresult *res = PQexec( m_conn, asSql.c_str() );
      if ( res == NULL )
      {
         setErrorInfo( DBI_MEMORY_ALLOC_ERROR, "Memory allocation error" );
         return NULL;
      }

      return new PgSQLRecordset( this, res );
   }

   int PgSQLConnection::close()
   {
      if ( m_conn == NULL )
      {
         // TODO: really err on this or just return good, the connection
         //       is not opened, so fail on close?
         return setErrorInfo( DBI_NO_CONNECTION_ERROR,
                              "No database connection" );
      }

      PQfinish( m_conn );
      m_conn = NULL;

      setErrorInfo( DBI_OK, "" );

      return m_errorCode;
   }

   PgSQLRecordset::PgSQLRecordset( PgSQLConnection *connClass,
                                   PGresult *res )
   {
      m_connClass = connClass;
      m_res = res;

      setErrorInfo( DBI_OK, "" );
      m_affectedRows = -1;
      m_rowCount = PQntuples( m_res );
      m_columnCount = PQnfields( m_res );
   }

   int PgSQLRecordset::columnIndex( const String *columnName )
   {
      if ( m_res == NULL )
      {
         setErrorInfo ( DBI_NO_RESULT_ERROR,  "No recordset opened" );
         return -1;
      }

      AutoCString asColumnName( columnName );
      return PQfnumber( m_res, asColumnName.c_str() );
   }

   int PgSQLRecordset::columnName( const int columnIndex, String &name )
   {
      if ( m_res == NULL )
      {
         return setErrorInfo( DBI_NO_RESULT_ERROR, "No recordset opened" );
      }

      char *columnName = PQfname( m_res, columnIndex );
      if ( columnName == NULL )
      {
         return setErrorInfo( DBI_COLUMN_INDEX_ERROR,
                              "Column index is out of range" );
      }

      name = columnName;
      return setErrorInfo( DBI_OK, "", false );
   }

   int PgSQLRecordset::value( const int columnIndex, String &value )
   {
      if ( m_res == NULL )
      {
         return setErrorInfo( DBI_NO_RESULT_ERROR, "No recordset opened" );
      }
      else if ( columnIndex >= m_columnCount || columnIndex < 0 )
      {
         return setErrorInfo( DBI_COLUMN_INDEX_ERROR,
                              "Column index is out of range" );
      }
      else if ( PQgetisnull( m_res, m_rowIndex, columnIndex ) == 1 )
      {
         return setErrorInfo( DBI_NULL_VALUE_WARNING, "Value is NULL" );
      }

      value = PQgetvalue( m_res, m_rowIndex, columnIndex );

      return setErrorInfo( DBI_OK, "" );
   }

   int PgSQLRecordset::next()
   {
      if ( m_res == NULL )
      {
         return setErrorInfo( DBI_NO_RESULT_ERROR, "No recordset opened" );
      }
      else if ( m_rowIndex >= m_rowCount )
      {
         return setErrorInfo( DBI_EOF_WARNING, "End of result set" );
      }

      m_rowIndex++;

      return setErrorInfo( DBI_OK, "" );
   }

   int PgSQLRecordset::close()
   {
      if ( m_res == NULL )
      {
         // TODO: really do this? See close comment on Connection class
         return setErrorInfo( DBI_NO_RESULT_ERROR, "No recordset opened" );
      }

      PQclear( m_res );
      m_res = NULL;

      return setErrorInfo( DBI_OK, "" );
   }
}

/* end of pgsql_mod.cpp */

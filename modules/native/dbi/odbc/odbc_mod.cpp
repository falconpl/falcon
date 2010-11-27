/*
   FALCON - The Falcon Programming Language.
   FILE: odbc_mod.cpp

   ODBC driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:23:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "odbc_mod.h"
#include <string.h>

namespace Falcon
{

/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

Sqlite3InBind::Sqlite3InBind( odbc_stmt* stmt ):
      DBIInBind(true),  // always changes binding
      m_stmt(stmt)
{}

Sqlite3InBind::~Sqlite3InBind()
{
   // nothing to do: the statement is not ours.
}


void Sqlite3InBind::onFirstBinding( int size )
{
   // nothing to allocate here.
}

void Sqlite3InBind::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];

   switch( item.type() )
   {
   // set to null
   case DBIBindItem::t_nil:
      odbc_bind_null( m_stmt, num+1 );
      break;

   case DBIBindItem::t_bool:
   case DBIBindItem::t_int:
      odbc_bind_int64( m_stmt, num+1, item.asInteger() );
      break;

   case DBIBindItem::t_double:
      odbc_bind_double( m_stmt, num+1, item.asDouble() );
      break;

   case DBIBindItem::t_string:
      odbc_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_STATIC );
      break;

   case DBIBindItem::t_buffer:
      odbc_bind_blob( m_stmt, num+1, item.asBuffer(), item.asStringLen(), SQLITE_STATIC );
      break;

   // the time has normally been decoded in the buffer
   case DBIBindItem::t_time:
      odbc_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_STATIC );
      break;
   }
}


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetODBC::DBIRecordsetODBC( DBIHandleODBC *dbh, odbc_stmt *res, const ItemArray& params )
    : DBIRecordset( dbh ),
      m_stmt( res ),
      m_bind( res )
{
   m_bAsString = dbh->options()->m_bFetchStrings;
   m_bind.bind( params );
   m_row = -1; // BOF
   m_columnCount = odbc_column_count( res );
}

DBIRecordsetODBC::~DBIRecordsetODBC()
{
   if ( m_stmt != NULL )
      close();
}

int DBIRecordsetODBC::getColumnCount()
{
   return m_columnCount;
}

int64 DBIRecordsetODBC::getRowIndex()
{
   return m_row;
}

int64 DBIRecordsetODBC::getRowCount()
{
   return -1; // we don't know
}


bool DBIRecordsetODBC::getColumnName( int nCol, String& name )
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   name.bufferize( odbc_column_name( m_stmt, nCol ) );

   return true;
}


bool DBIRecordsetODBC::fetchRow()
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   int res = odbc_step( m_stmt );

   if( res == SQLITE_DONE )
      return false;
   else if ( res != SQLITE_ROW )
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_FETCH, res );

   // more data incoming
   m_row++;
   return true;
}


bool DBIRecordsetODBC::discard( int64 ncount )
{
   while ( ncount > 0 )
   {
      if( ! fetchRow() )
      {
         return false;
      }
      --ncount;
   }

   return true;
}


void DBIRecordsetODBC::close()
{
   if( m_stmt != 0 )
   {
      odbc_finalize( m_stmt );
      m_stmt = 0;
   }
}

bool DBIRecordsetODBC::getColumnValue( int nCol, Item& value )
{
   if( m_stmt == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   switch ( odbc_column_type(m_stmt, nCol) )
   {
   case SQLITE_NULL:
      value.setNil();
      return true;

   case SQLITE_INTEGER:
      if( m_bAsString )
      {
         value = new CoreString( (const char*)odbc_column_text(m_stmt, nCol), -1 );
      }
      else
      {
         value.setInteger( odbc_column_int64(m_stmt, nCol) );
      }
      return true;

   case SQLITE_FLOAT:
      if( m_bAsString )
      {
         value = new CoreString( (const char*)odbc_column_text( m_stmt, nCol ), -1 );
      }
      else
      {
         value.setNumeric( odbc_column_double( m_stmt, nCol ) );
      }
      return true;

   case SQLITE_BLOB:
      {
         int len =  odbc_column_bytes( m_stmt, nCol );
         MemBuf* mb = new MemBuf_1( len );
         memcpy( mb->data(), (byte*) odbc_column_blob( m_stmt, nCol ), len );
         value = mb;
      }
      return true;


   case SQLITE_TEXT:
      {
         CoreString* cs = new CoreString;
         cs->fromUTF8( (const char*) odbc_column_text( m_stmt, nCol ) );
         value = cs;
      }
      return true;
   }

   return false;
}


/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementODBC::DBIStatementODBC( DBIHandleODBC *dbh, odbc_stmt* stmt ):
   DBIStatement( dbh ),
   m_statement( stmt ),
   m_inBind( stmt )
{
}

DBIStatementODBC::~DBIStatementODBC()
{
   close();
}

int64 DBIStatementODBC::execute( const ItemArray& params )
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   m_inBind.bind(params);
   int res = odbc_step( m_statement );
   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_EXEC, res );
   }

   // SQLite doesn't distinguish between fetch and insert statements; we do.
   // Exec never returns a recordset; instead, it is used to insert and re-issue
   // repeatedly statemnts. This is accomplished by Sqllite by issuing a reset
   // after each step.
   res = odbc_reset( m_statement );
   if( res != SQLITE_OK )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_EXEC, res );
   }

   return 0;
}

void DBIStatementODBC::reset()
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   int res = odbc_reset( m_statement );
   if( res != SQLITE_OK )
   {
      DBIHandleODBC::throwError( FALCON_DBI_ERROR_RESET, res );
   }
}

void DBIStatementODBC::close()
{
   if( m_statement != 0 )
   {
      odbc_finalize( m_statement );
      m_statement = 0;
   }
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleODBC::DBIHandleODBC():
   m_bInTrans(false)
{
   m_conn = NULL;
}

DBIHandleODBC::DBIHandleODBC( odbc *conn ):
   m_bInTrans(false)
{
   m_conn = conn;
}

DBIHandleODBC::~DBIHandleODBC()
{
   close();
}

void DBIHandleODBC::options( const String& params )
{
   if( m_settings.parse( params ) )
   {
      // To turn off the autocommit.
      SQLSetConnectAttr( m_conn->m_hHdbc, SQL_AUTOCOMMIT, 
            m_settings.m_bAutocommit ? SQL_AUTOCOMMIT_ON: SQL_AUTOCOMMIT_OFF, 
            0 );
   }
   else
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
   }
}

const DBISettingParams* DBIHandleODBC::options() const
{
   return &m_settings;
}

DBIRecordset *DBIHandleODBC::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   odbc_stmt* pStmt = int_prepare( sql );
   int count = odbc_column_count( pStmt );
   if( count == 0 )
   {
      throw new DBIError( ErrorParam(FALCON_DBI_ERROR_QUERY_EMPTY, __LINE__ ) );
   }
   affectedRows = -1;

   // the bindings must stay with the recordset...
   return new DBIRecordsetODBC( this, pStmt, params );
}

void DBIHandleODBC::perform( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   odbc_stmt* pStmt = int_prepare( sql );
   int_execute( pStmt, params );
   affectedRows = odbc_changes( m_conn );
}


DBIRecordset* DBIHandleODBC::call( const String &sql, int64 &affectedRows, const ItemArray& params )
{

   odbc_stmt* pStmt = int_prepare( sql );
   int count = odbc_column_count( pStmt );
   if( count == 0 )
   {
      int_execute( pStmt, params );
      affectedRows = odbc_changes( m_conn );
      return 0;
   }
   else
   {
      // the bindings must stay with the recordset...
      return new DBIRecordsetODBC( this, pStmt, params );
   }
}


DBIStatement* DBIHandleODBC::prepare( const String &query )
{
   odbc_stmt* pStmt = int_prepare( query );
   return new DBIStatementODBC( this, pStmt );
}


odbc_stmt* DBIHandleODBC::int_prepare( const String &sql ) const
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   AutoCString zSql( sql );
   odbc_stmt* pStmt = 0;
   int res = odbc_prepare_v2( m_conn, zSql.c_str(), zSql.length(), &pStmt, 0 );
   if( res != SQLITE_OK )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }

   return pStmt;
}

void DBIHandleODBC::int_execute( odbc_stmt* pStmt, const ItemArray& params )
{
   // int_execute is NEVER called alone
   fassert( m_conn != 0 );

   int res;
   if( params.length() > 0 )
   {
      Sqlite3InBind binds( pStmt );
      binds.bind(params);
      res = odbc_step( pStmt );
      odbc_finalize( pStmt );
   }
   else
   {
      res = odbc_step( pStmt );
      odbc_finalize( pStmt );
   }

   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }
}


void DBIHandleODBC::begin()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( !m_bInTrans )
   {
      char* error;
      int res = odbc_exec( m_conn, "BEGIN TRANSACTION", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = true;
   }
}

void DBIHandleODBC::commit()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   // Sqlite doesn't ignore COMMIT out of transaction.
   // We do; so we must filter them
   if( m_bInTrans )
   {
      char* error;
      int res = odbc_exec( m_conn, "COMMIT", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = false;
   }
}


void DBIHandleODBC::rollback()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   // Sqlite doesn't ignore COMMIT out of transaction.
   // We do; so we must filter them
   if( m_bInTrans )
   {
      char* error;
      int res = odbc_exec( m_conn, "ROLLBACK", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = false;
   }
}


void DBIHandleODBC::selectLimited( const String& query,
      int64 nBegin, int64 nCount, String& result )
{
   String sBegin, sCount;

   if ( nBegin > 0 )
   {
      sBegin = " OFFSET ";
      sBegin.N( nBegin );
   }

   if( nCount > 0 )
   {
      sCount.N( nCount );
   }

   result = "SELECT " + query;

   if( nCount != 0 || nBegin != 0 )
   {
      result += " LIMIT " + sCount + sBegin;
   }
}


void DBIHandleODBC::throwError( int falconError, int sql3Error, char* edesc )
{
   String err = String("(").N(sql3Error).A(") ");
   if( edesc == 0 )
      err += errorDesc( sql3Error );
   else
   {
      err.A(edesc);
      err.bufferize();
      odbc_free( edesc ); // got from odbc_malloc, must be freed
   }

   throw new DBIError( ErrorParam(falconError, __LINE__ )
         .extra(err) );
}


String DBIHandleODBC::errorDesc( int error )
{
   switch( error & 0xFF )
   {
   case SQLITE_OK           : return "Successful result";
   case SQLITE_ERROR        : return "SQL error or missing database";
   case SQLITE_INTERNAL     : return "Internal logic error in SQLite";
   case SQLITE_PERM         : return "Access permission denied";
   case SQLITE_ABORT        : return "Callback routine requested an abort";
   case SQLITE_BUSY         : return "The database file is locked";
   case SQLITE_LOCKED       : return "A table in the database is locked";
   case SQLITE_NOMEM        : return "A malloc() failed";
   case SQLITE_READONLY     : return "Attempt to write a readonly database";
   case SQLITE_INTERRUPT    : return "Operation terminated by odbc_interrupt()";
   case SQLITE_IOERR        : return "Some kind of disk I/O error occurred";
   case SQLITE_CORRUPT      : return "The database disk image is malformed";
   case SQLITE_NOTFOUND     : return "NOT USED. Table or record not found";
   case SQLITE_FULL         : return "Insertion failed because database is full";
   case SQLITE_CANTOPEN     : return "Unable to open the database file";
   case SQLITE_PROTOCOL     : return "NOT USED. Database lock protocol error";
   case SQLITE_EMPTY        : return "Database is empty";
   case SQLITE_SCHEMA       : return "The database schema changed";
   case SQLITE_TOOBIG       : return "String or BLOB exceeds size limit";
   case SQLITE_CONSTRAINT   : return "Abort due to constraint violation";
   case SQLITE_MISMATCH     : return "Data type mismatch";
   case SQLITE_MISUSE       : return "Library used incorrectly";
   case SQLITE_NOLFS        : return "Uses OS features not supported on host";
   case SQLITE_AUTH         : return "Authorization denied";
   case SQLITE_FORMAT       : return "Auxiliary database format error";
   case SQLITE_RANGE        : return "2nd parameter to odbc_bind out of range";
   case SQLITE_NOTADB       : return "File opened that is not a database file";
   case SQLITE_ROW          : return "odbc_step() has another row ready";
   case SQLITE_DONE         : return "odbc_step() has finished executing";
   }

   return "Unknown error";
/*
   case SQLITE_IOERR_READ              (SQLITE_IOERR | (1<<8))
   case SQLITE_IOERR_SHORT_READ        (SQLITE_IOERR | (2<<8))
   case SQLITE_IOERR_WRITE             (SQLITE_IOERR | (3<<8))
   case SQLITE_IOERR_FSYNC             (SQLITE_IOERR | (4<<8))
   case SQLITE_IOERR_DIR_FSYNC         (SQLITE_IOERR | (5<<8))
   case SQLITE_IOERR_TRUNCATE          (SQLITE_IOERR | (6<<8))
   case SQLITE_IOERR_FSTAT             (SQLITE_IOERR | (7<<8))
   case SQLITE_IOERR_UNLOCK            (SQLITE_IOERR | (8<<8))
   case SQLITE_IOERR_RDLOCK            (SQLITE_IOERR | (9<<8))
   case SQLITE_IOERR_DELETE            (SQLITE_IOERR | (10<<8))
   case SQLITE_IOERR_BLOCKED           (SQLITE_IOERR | (11<<8))
   case SQLITE_IOERR_NOMEM             (SQLITE_IOERR | (12<<8))
   case SQLITE_IOERR_ACCESS            (SQLITE_IOERR | (13<<8))
   case SQLITE_IOERR_CHECKRESERVEDLOCK (SQLITE_IOERR | (14<<8))
   case SQLITE_IOERR_LOCK              (SQLITE_IOERR | (15<<8))
   case SQLITE_IOERR_CLOSE             (SQLITE_IOERR | (16<<8))
   case SQLITE_IOERR_DIR_CLOSE         (SQLITE_IOERR | (17<<8))
   case SQLITE_LOCKED_SHAREDCACHE      (SQLITE_LOCKED | (1<<8) )
*/

}

int64 DBIHandleODBC::getLastInsertedId( const String& )
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   return odbc_last_insert_rowid( m_conn );
}


void DBIHandleODBC::close()
{
   if ( m_conn != NULL )
   {
      if( m_bInTrans )
      {
         odbc_exec( m_conn, "ROLLBACK", 0, 0, 0 );
         m_bInTrans = false;
      }

      odbc_close( m_conn );
      m_conn = NULL;
   }
}

}

/* end of odbc_mod.cpp */


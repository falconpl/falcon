/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_mod.cpp

   SQLite3 driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:23:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "sqlite3_mod.h"
#include <string.h>

namespace Falcon
{

/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

Sqlite3InBind::Sqlite3InBind( sqlite3_stmt* stmt ):
      DBIInBind(true),  // always changes binding
      m_stmt(stmt)
{}

Sqlite3InBind::~Sqlite3InBind()
{
   // nothing to do: the statement is not ours.
}


void Sqlite3InBind::onFirstBinding( int )
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
      sqlite3_bind_null( m_stmt, num+1 );
      break;

   case DBIBindItem::t_bool:
   case DBIBindItem::t_int:
      sqlite3_bind_int64( m_stmt, num+1, item.asInteger() );
      break;

   case DBIBindItem::t_double:
      sqlite3_bind_double( m_stmt, num+1, item.asDouble() );
      break;

   case DBIBindItem::t_string:
      //TODO: Here, we could use SQLITE_STATIC for everything except for queries.
      //That's because sqlite wants the variable binding to stay valid while it
      //fetches each new record, as it doesn't create the recordset when the query
      //is launched -- so, we must let SQLite to do its own copy in queries,
      //but for everything else, the m_ibind storage would be enough. We should
      //optimize this function so that it relies to SQLITE_TRANSIENT only in queries.

      sqlite3_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_TRANSIENT );
      break;

   case DBIBindItem::t_buffer:
      sqlite3_bind_blob( m_stmt, num+1, item.asBuffer(), item.asStringLen(), SQLITE_TRANSIENT );
      break;

   // the time has normally been decoded in the buffer
   case DBIBindItem::t_time:
      sqlite3_bind_text( m_stmt, num+1, item.asString(), item.asStringLen(), SQLITE_TRANSIENT );
      break;
   }
}


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetSQLite3::DBIRecordsetSQLite3( DBIHandleSQLite3 *dbh, sqlite3_stmt *res )
    : DBIRecordset( dbh ),
      m_stmt( res )
{
   m_bAsString = dbh->options()->m_bFetchStrings;
   m_row = -1; // BOF
   m_columnCount = sqlite3_column_count( res );
}


DBIRecordsetSQLite3::~DBIRecordsetSQLite3()
{
   if ( m_stmt != NULL )
      close();
}

int DBIRecordsetSQLite3::getColumnCount()
{
   return m_columnCount;
}

int64 DBIRecordsetSQLite3::getRowIndex()
{
   return m_row;
}

int64 DBIRecordsetSQLite3::getRowCount()
{
   return -1; // we don't know
}


bool DBIRecordsetSQLite3::getColumnName( int nCol, String& name )
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   name.bufferize( sqlite3_column_name( m_stmt, nCol ) );

   return true;
}


bool DBIRecordsetSQLite3::fetchRow()
{
   if( m_stmt == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   int res = sqlite3_step( m_stmt );

   if( res == SQLITE_DONE )
      return false;
   else if ( res != SQLITE_ROW )
      DBIHandleSQLite3::throwError( FALCON_DBI_ERROR_FETCH, res );

   // more data incoming
   m_row++;
   return true;
}


bool DBIRecordsetSQLite3::discard( int64 ncount )
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


void DBIRecordsetSQLite3::close()
{
   if( m_stmt != 0 )
   {
      sqlite3_finalize(m_stmt);
      m_stmt = 0;
   }
}

bool DBIRecordsetSQLite3::getColumnValue( int nCol, Item& value )
{
   if( m_stmt == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   if ( nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   switch ( sqlite3_column_type(m_stmt, nCol) )
   {
   case SQLITE_NULL:
      value.setNil();
      return true;

   case SQLITE_INTEGER:
      if( m_bAsString )
      {
         value = FALCON_GC_HANDLE(new String( (const char*)sqlite3_column_text(m_stmt, nCol), -1 ));
      }
      else
      {
         value.setInteger( sqlite3_column_int64(m_stmt, nCol) );
      }
      return true;

   case SQLITE_FLOAT:
      if( m_bAsString )
      {
         value =  FALCON_GC_HANDLE(new String( (const char*)sqlite3_column_text( m_stmt, nCol ), -1 ));
      }
      else
      {
         value.setNumeric( sqlite3_column_double( m_stmt, nCol ) );
      }
      return true;

   case SQLITE_BLOB:
      {
         int len =  sqlite3_column_bytes( m_stmt, nCol );
         String* sVal = new String();
         sVal->reserve(len);
         sVal->toMemBuf();
         memcpy( sVal->getRawStorage(), (byte*) sqlite3_column_blob( m_stmt, nCol ), len );
         sVal->size(len);

         value = FALCON_GC_HANDLE(sVal);
      }
      return true;


   case SQLITE_TEXT:
      {
         String* cs = new String;
         cs->fromUTF8( (const char*) sqlite3_column_text( m_stmt, nCol ) );
         value = FALCON_GC_HANDLE(cs);
      }
      return true;
   }

   return false;
}


/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementSQLite3::DBIStatementSQLite3( DBIHandleSQLite3 *dbh, sqlite3_stmt* stmt ):
   DBIStatement( dbh ),
   m_statement( stmt ),
   m_inBind( stmt ),
   m_bFirst( false )
{
}


DBIStatementSQLite3::~DBIStatementSQLite3()
{
   close();
}

DBIRecordset* DBIStatementSQLite3::execute( ItemArray* params )
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   // clear previous recordset
   int res;

   if ( m_bFirst )
   {
      m_bFirst = false;
   }
   else
   {
      res = sqlite3_reset( m_statement );
      if( res != SQLITE_OK )
      {
         DBIHandleSQLite3::throwError( FALCON_DBI_ERROR_EXEC, res );
      }
   }

   if( params != 0 )
   {
      m_inBind.bind(*params);
   }
   else 
   {
      m_inBind.unbind();
   }
   
   res = sqlite3_step( m_statement );
   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      DBIHandleSQLite3::throwError( FALCON_DBI_ERROR_EXEC, res );
   }

/*   
   if ( sqlite3_column_count( m_statement ) != 0 )
   {
      // we do have a recorset
      return new DBIRecordsetSQLite3( static_cast<DBIHandleSQLite3*>( m_dbh ), m_pStmt );
   }
*/
   return 0;
}

void DBIStatementSQLite3::reset()
{
   if( m_statement == 0 )
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   int res = sqlite3_reset( m_statement );
   if( res != SQLITE_OK )
   {
      DBIHandleSQLite3::throwError( FALCON_DBI_ERROR_RESET, res );
   }
}

void DBIStatementSQLite3::close()
{
   if( m_statement != 0 )
   {
      sqlite3_finalize( m_statement );
      m_statement = 0;
   }
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleSQLite3::DBIHandleSQLite3( const Class* h ):
      DBIHandle(h),
      m_bInTrans(false)
{
   m_conn = NULL;
}

DBIHandleSQLite3::DBIHandleSQLite3( const Class* h, sqlite3 *conn ):
      DBIHandle(h),
      m_bInTrans(false)
{
   m_conn = conn;
   sqlite3_extended_result_codes( conn, 1 );
}

DBIHandleSQLite3::~DBIHandleSQLite3()
{
   close();
}


void DBIHandleSQLite3::connect( const String& parameters )
{
   // Parse the connection string.
   DBIConnParams connParams;

   if( ! connParams.parse( parameters ) || connParams.m_szDb == 0 )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .extra( parameters )
      );
   }

   int flags = SQLITE_OPEN_READWRITE;
   if( connParams.m_sCreate == "always" )
   {
      flags |= SQLITE_OPEN_CREATE;

      // sqlite3 doesn't drop databases: delete files.
      String sURI(connParams.m_szDb);
      if ( Engine::instance()->vfs().fileType( sURI, true ) == FileStat::_normal )
      {
         try
         {
            Engine::instance()->vfs().erase( sURI );
         }
         catch (Error* e)
         {
            e->decref();
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT_CREATE, __LINE__)
                      .extra( parameters )
                   );
         }
      }
   }
   else if ( connParams.m_sCreate == "cond" )
   {
      flags |= SQLITE_OPEN_CREATE;
   }
   else if( connParams.m_sCreate != "" )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
              .extra( parameters )
           );
   }

   sqlite3 *conn;
   int result = sqlite3_open_v2( connParams.m_szDb, &conn, flags, NULL );

   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
   }
   else if ( result == SQLITE_CANTOPEN )
   {
      int er = connParams.m_sCreate == "cond" ?
               FALCON_DBI_ERROR_CONNECT_CREATE : FALCON_DBI_ERROR_DB_NOTFOUND;

      throw new DBIError( ErrorParam( er, __LINE__)
                    .extra( sqlite3_errmsg( conn ) )
                 );
   }
   else if ( result != SQLITE_OK )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
              .extra( sqlite3_errmsg( conn ) )
           );
   }

   m_conn = conn;
}

void DBIHandleSQLite3::options( const String& params )
{
   if( m_settings.parse( params ) )
   {
      // To turn off the autocommit.
      if( ! m_settings.m_bAutocommit )
      {
         begin();
      }
   }
   else
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
   }
}

const DBISettingParams* DBIHandleSQLite3::options() const
{
   return &m_settings;
}

DBIRecordset *DBIHandleSQLite3::query( const String &sql, ItemArray* params )
{
   sqlite3_stmt* pStmt = int_prepare( sql );
   
   int res;
   if( params != 0 )
   {
      Sqlite3InBind binds( pStmt );
      binds.bind(*params);
      res = sqlite3_step( pStmt );
   }
   else
   {
      res = sqlite3_step( pStmt );
   }

   if( res != SQLITE_OK
         && res != SQLITE_DONE
         && res != SQLITE_ROW )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }

   // do we have a recordset?
   int count = sqlite3_column_count( pStmt );
   m_nLastAffected = sqlite3_changes( m_conn );
   if( count == 0 )
   {
      sqlite3_finalize( pStmt );
      return 0;
   }
   else
   {
      sqlite3_reset( pStmt );
      // the bindings must stay with the recordset...
      return new DBIRecordsetSQLite3( this, pStmt );
   }
}


DBIStatement* DBIHandleSQLite3::prepare( const String &query )
{
   sqlite3_stmt* pStmt = int_prepare( query );
   return new DBIStatementSQLite3( this, pStmt );
}


sqlite3_stmt* DBIHandleSQLite3::int_prepare( const String &sql ) const
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   AutoCString zSql( sql );
   sqlite3_stmt* pStmt = 0;
   int res = sqlite3_prepare_v2( m_conn, zSql.c_str(), zSql.length(), &pStmt, 0 );
   if( res != SQLITE_OK )
   {
      throwError( FALCON_DBI_ERROR_QUERY, res );
   }

   return pStmt;
}



void DBIHandleSQLite3::begin()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( !m_bInTrans )
   {
      char* error;
      int res = sqlite3_exec( m_conn, "BEGIN TRANSACTION", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = true;
   }
}

void DBIHandleSQLite3::commit()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   // Sqlite doesn't ignore COMMIT out of transaction.
   // We do; so we must filter them
   if( m_bInTrans )
   {
      char* error;
      int res = sqlite3_exec( m_conn, "COMMIT", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = false;
   }
}


void DBIHandleSQLite3::rollback()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   // Sqlite doesn't ignore COMMIT out of transaction.
   // We do; so we must filter them
   if( m_bInTrans )
   {
      char* error;
      int res = sqlite3_exec( m_conn, "ROLLBACK", 0, 0, &error );
      if( res != 0 )
         throwError( FALCON_DBI_ERROR_TRANSACTION, res, error );
      m_bInTrans = false;
   }
}


void DBIHandleSQLite3::selectLimited( const String& query,
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


void DBIHandleSQLite3::throwError( int falconError, int sql3Error, char* edesc )
{
   String err = String("(").N(sql3Error).A(") ");
   if( edesc == 0 )
      err += errorDesc( sql3Error );
   else
   {
      err.A(edesc);
      err.bufferize();
      sqlite3_free( edesc ); // got from sqlite3_malloc, must be freed
   }

   throw new DBIError( ErrorParam(falconError, __LINE__ )
         .extra(err) );
}


String DBIHandleSQLite3::errorDesc( int error )
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
   case SQLITE_INTERRUPT    : return "Operation terminated by sqlite3_interrupt()";
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
   case SQLITE_RANGE        : return "2nd parameter to sqlite3_bind out of range";
   case SQLITE_NOTADB       : return "File opened that is not a database file";
   case SQLITE_ROW          : return "sqlite3_step() has another row ready";
   case SQLITE_DONE         : return "sqlite3_step() has finished executing";
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

int64 DBIHandleSQLite3::getLastInsertedId( const String& )
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   return sqlite3_last_insert_rowid( m_conn );
}


void DBIHandleSQLite3::close()
{
   if ( m_conn != NULL )
   {
      if( m_bInTrans )
      {
         sqlite3_exec( m_conn, "COMMIT", 0, 0, 0 );
         m_bInTrans = false;
      }

      sqlite3_close(m_conn);
      m_conn = NULL;
   }
}

}

/* end of sqlite3_mod.cpp */


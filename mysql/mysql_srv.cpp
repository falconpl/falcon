/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_srv.cpp
 *
 * MySQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <errmsg.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "mysql_mod.h"
#include "dbi_mod.h"
#include "dbi_st.h"

namespace Falcon
{

class DBITimeConverter_MYSQL_TIME: public DBITimeConverter
{
public:
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
} DBITimeConverter_MYSQL_TIME_impl;

void DBITimeConverter_MYSQL_TIME::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( bufsize >= sizeof( MYSQL_TIME ) );

   MYSQL_TIME* mtime = (MYSQL_TIME*) buffer;
   mtime->year = (unsigned int) ts->m_year;
   mtime->month = (unsigned int) ts->m_month;
   mtime->day = (unsigned int) ts->m_day;
   mtime->hour = (unsigned int) ts->m_hour;
   mtime->minute = (unsigned int) ts->m_minute;
   mtime->second = (unsigned int) ts->m_second;
   mtime->second_part = (unsigned int) ts->m_msec;
   mtime->neg = 0;

   bufsize = sizeof( MYSQL_TIME );
}


/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

DBIBindMySQL::DBIBindMySQL():
      m_mybind(0)
{}

DBIBindMySQL::~DBIBindMySQL()
{
   memFree( m_mybind );
}


void DBIBindMySQL::onFirstBinding( int size )
{
   m_mybind = (MYSQL_BIND*) memAlloc( sizeof(MYSQL_BIND) * size );
   memset( m_mybind, 0, sizeof(MYSQL_BIND) * size );
}


void DBIBindMySQL::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];
   MYSQL_BIND& myitem = m_mybind[num];

   switch( item.type() )
   {
   // set to null
   case DBIBindItem::t_nil:
      myitem.buffer_type = MYSQL_TYPE_NULL;
      *((my_bool*) item.data()) = 1;
      break;

   case DBIBindItem::t_bool:
      myitem.buffer_type = MYSQL_TYPE_BIT;
      myitem.buffer = item.asBoolPtr();
      myitem.buffer_length = 1;
      break;

   case DBIBindItem::t_int:
      myitem.buffer_type = MYSQL_TYPE_LONGLONG;
      myitem.buffer = item.asIntegerPtr();
      myitem.buffer_length = sizeof( int64 );
      break;

   case DBIBindItem::t_double:
      myitem.buffer_type = MYSQL_TYPE_DOUBLE;
      myitem.buffer = item.asDoublePtr();
      myitem.buffer_length = sizeof( double );
      break;

    case DBIBindItem::t_string:
         myitem.buffer_type = MYSQL_TYPE_STRING;
         myitem.buffer = (void*) item.asString();
         myitem.buffer_length = item.asStringLen();
         break;

    case DBIBindItem::t_buffer:
         myitem.buffer_type = MYSQL_TYPE_BLOB;
         myitem.buffer = item.asBuffer();
         myitem.buffer_length = item.asStringLen();
         break;

    case DBIBindItem::t_time:
         myitem.buffer_type = MYSQL_TYPE_TIMESTAMP;
         myitem.buffer = item.databuffer();
         myitem.buffer_length = sizeof( MYSQL_TIME );
         break;
   }
}



/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetMySQL::DBIRecordsetMySQL( DBITransaction *dbt, MYSQL_RES *res, MYSQL_STMT *stmt )
    : DBIRecordset( dbt ),
      m_res( res ),
      m_stmt( stmt )
{
   m_row = -1; // BOF
   m_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
   m_columnCount = mysql_num_fields( res );
   m_fields = mysql_fetch_fields( res );
}

DBIRecordsetMySQL::~DBIRecordsetMySQL()
{
   if ( m_res != NULL )
      close();
}

/*
dbi_type DBIRecordsetMySQL::getFalconType( int typ )
{
   switch ( typ )
   {
   case MYSQL_TYPE_TINY:
   case MYSQL_TYPE_SHORT:
   case MYSQL_TYPE_LONG:
   case MYSQL_TYPE_INT24:
   case MYSQL_TYPE_BIT:
   case MYSQL_TYPE_YEAR:
      return dbit_integer;

   case MYSQL_TYPE_LONGLONG:
      return dbit_integer64;

   case MYSQL_TYPE_DECIMAL:
   case MYSQL_TYPE_NEWDECIMAL:
   case MYSQL_TYPE_FLOAT:
   case MYSQL_TYPE_DOUBLE:
      return dbit_numeric;

   case MYSQL_TYPE_DATE:
      return dbit_date;

   case MYSQL_TYPE_TIME:
      return dbit_time;

   case MYSQL_TYPE_DATETIME: // TODO: MYSQL_TYPE_TIMESTAMP ?!?
      return dbit_datetime;

   default:
      return dbit_string;
   }
}
*/

int DBIRecordsetMySQL::getColumnCount()
{
   return m_columnCount;
}

bool DBIRecordsetMySQL::getColumnName( int nCol, String& name )
{
   if( nCol >=0  && nCol < m_columnCount )
   {
      name.fromUTF8( m_fields[nCol].name );
      return true;
   }
   return false;
}


bool DBIRecordsetMySQL::getColumnValue( int nCol, Item& value )
{
   if ( m_row == 0 || nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }
#if 0 // TODO
   switch( m_fields[nCol].type )
   {

   }

   value.fromUTF8( m_rowData[columnIndex] );
#endif
   return true;
}


int64 DBIRecordsetMySQL::getRowCount()
{
   return m_rowCount;
}


int64 DBIRecordsetMySQL::getRowIndex()
{
   return m_row;
}

void DBIRecordsetMySQL::close()
{
   if ( m_res != NULL ) {
      mysql_free_result( m_res );
      m_res = NULL;
   }

   if ( m_stmt != 0 )
   {
      mysql_stmt_close( m_stmt );
      m_stmt = 0;
   }
}


/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionMySQL::DBITransactionMySQL( DBIHandle *dbh, DBISettingParams* settings ):
      DBITransaction( dbh, settings ),
      m_statement(0)
{
}


DBITransactionMySQL::~DBITransactionMySQL()
{
   if ( m_statement != 0 )
   {
      mysql_stmt_close( m_statement );
      m_statement = 0;
   }

   delete m_inBind;
}


DBIRecordset *DBITransactionMySQL::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // prepare and execute -- will create a new m_statement
   prepare( sql );
   execute( params, affectedRows );

   // We want a result recordset
   MYSQL_RES* meta = mysql_stmt_result_metadata( m_statement );
   if( meta == 0 )
   {
      // the query didn't return a recorset
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY_EMPTY );
   }
   else
   {
      DBIRecordsetMySQL* recset = new DBIRecordsetMySQL( this, meta, m_statement );

      // ok. Do the user wanted all the result back?
      if( m_settings->m_nPrefetch < 0 )
      {
         if( ! mysql_stmt_store_result( m_statement ) )
         {
            m_statement = 0;
            delete recset;
            mysql_free_result( meta );
            getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
         }
      }

      // pass the ownership of the statement to the recordset.
      m_statement = 0;
      return recset;
   }

   return 0; // to make the compiler happy
}


void DBITransactionMySQL::call( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // if we don't have var params, we can use the standard query, after proper escape.
   if ( params.length() == 0 )
   {
      MYSQL *conn = getMySql()->getConn();

      AutoCString asQuery( sql );
      if( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
      }
   }
   else
   {
     // prepare and execute
     prepare( sql );
     execute( params, affectedRows );
   }
}


void DBITransactionMySQL::prepare( const String &query )
{
   delete m_inBind;
   m_inBind = 0;

   // setup the statement.
   if ( m_statement != 0 )
   {
      mysql_stmt_close( m_statement );
   }

   m_statement = mysql_stmt_init( getMySql()->getConn() );
   if( m_statement == 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_NOMEM );
   }

   AutoCString cquery( query );
   if( mysql_stmt_prepare( m_statement, cquery.c_str(), cquery.length() ) != 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
   }

   // prepare the attributes as suggested by our parameters.
   unsigned long setting = m_settings->m_nCursorThreshold == 0  ?
         CURSOR_TYPE_READ_ONLY : CURSOR_TYPE_NO_CURSOR;

   mysql_stmt_attr_set( m_statement, STMT_ATTR_CURSOR_TYPE, &setting );

   if( m_settings->m_nPrefetch > 0 )
   {
      setting = (unsigned long) m_settings->m_nPrefetch;
      mysql_stmt_attr_set( m_statement, STMT_ATTR_PREFETCH_ROWS, &setting );
   }
   else if ( m_settings->m_nPrefetch == -1 )
   {
      setting = (unsigned long) 0xFFFFFFFF;
      mysql_stmt_attr_set( m_statement, STMT_ATTR_PREFETCH_ROWS, &setting );
   }
}


void DBITransactionMySQL::execute( const ItemArray& params, int64 &affectedRows )
{
   if ( m_statement == 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNPREP_EXEC );
   }

   // should we bind with the statement?
   if ( m_inBind == 0 )
   {
      if( params.length() != mysql_stmt_param_count( m_statement ) )
      {
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
      }

      // Do we have some parameter to bind?
      if( params.length() > 0 )
      {
         // params.lengh() == 0 is possible with totally static selects,
         // or other statements that will be run usually just once.
         // Inserts or other repetitive statements will have at least 1, so
         // this branch won't be repeatedly checked in the fast path.
         m_inBind = new DBIBindMySQL;
         m_inBind->bind( params );

         if( mysql_stmt_bind_param( m_statement, m_inBind->mybindings() ) != 0 )
         {
            getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_MIX );
         }
      }
   }
   else
   {
      m_inBind->bind( params );
   }

   if( mysql_stmt_execute( m_statement ) != 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_EXEC );
   }

   // row count?
   affectedRows = mysql_stmt_affected_rows( m_statement );
}



void DBITransactionMySQL::begin()
{
   int64 dummy;
   ItemArray arr;
   call( "BEGIN", dummy, arr );
   MYSQL *conn = ((DBIHandleMySQL *) m_dbh)->getConn();
   mysql_autocommit( conn, m_settings->m_bAutocommit );
   m_inTransaction = true;
}




void DBITransactionMySQL::commit()
{
   int64 dummy;
   ItemArray arr;
   call( "COMMIT", dummy, arr );
   m_inTransaction = false;
}

void DBITransactionMySQL::rollback()
{
   int64 dummy;
   ItemArray arr;
   call( "ROLLBACK", dummy, arr );
   m_inTransaction = false;
}

void DBITransactionMySQL::close()
{
   if ( m_inTransaction )
      commit();

   m_inTransaction = false;
}

int64 DBITransactionMySQL::getLastInsertedId( const String& name )
{
    // TODO
    return -1;
}


DBITransaction *DBITransactionMySQL::startTransaction( const String& settings )
{
   throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NO_SUBTRANS, __LINE__ ) );
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/
DBIHandleMySQL::~DBIHandleMySQL()
{
   DBIHandleMySQL::close();
}

bool DBIHandleMySQL::setTransOpt( const String& params )
{
   return m_settings.parse( params );
}

const DBISettingParams* DBIHandleMySQL::transOpt() const
{
   return &m_settings;
}

DBITransaction *DBIHandleMySQL::startTransaction( const String &options )
{
   DBITransactionMySQL* t;

   if ( options == "" )
   {
      t = new DBITransactionMySQL( this, new DBISettingParams( m_settings ) );
   }
   else
   {
      DBISettingParams* settings = new DBISettingParams;
      if (! settings->parse( options ) )
      {
         delete settings;
         throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS ).extra( options ) );
      }

      t = new DBITransactionMySQL( this, settings );
   }

   try
   {
      t->begin();
   }
   catch(...)
   {
      delete t;
      throw;
   }

   return t;
}

DBIHandleMySQL::DBIHandleMySQL()
{
   m_conn = NULL;
}

DBIHandleMySQL::DBIHandleMySQL( MYSQL *conn )
{
   m_conn = conn;
   // we'll be using UTF-8 charset
   mysql_set_character_set( m_conn, "utf8" );
}

#if 0
int64 DBIHandleMySQL::getLastInsertedId( const String& sequenceName )
{
   return mysql_insert_id( m_conn );
}
#endif

void DBIHandleMySQL::close()
{
   if ( m_conn != NULL ) {
      mysql_close( m_conn );
      m_conn = NULL;
   }
}

void DBIHandleMySQL::throwError( const char* file, int line, int code )
{
   const char *errorMessage = mysql_error( m_conn );
   String extra; // dummy

   if ( errorMessage != NULL )
   {
      String description;
      description.N( (int64) mysql_errno( m_conn ) ).A(": ");
      description.A( errorMessage );
      dbh_throwError( file, line, code, extra );
   }
   else
   {
      dbh_throwError( file, line, code, extra );
   }
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceMySQL::init()
{
}

DBIHandle *DBIServiceMySQL::connect( const String &parameters, bool persistent )
{
   // Parse the connection string.
   DBIConnParams connParams;

   // add MySQL specific parameters
   String sSocket, sFlags;
   const char *szSocket;
   connParams.addParameter( "socket", sSocket, &szSocket );
   connParams.addParameter( "flags", sFlags );

   MYSQL *conn = mysql_init( NULL );

   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
   }

   if( ! connParams.parse( parameters ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .extra( parameters )
      );
   }

   //long szFlags = 0;
   // TODO parse flags

   if ( mysql_real_connect( conn,
         connParams.m_szHost,
         connParams.m_szUser,
         connParams.m_szPassword,
         connParams.m_szDb,
         connParams.m_szPort == 0 ? 0 : atoi( connParams.m_szPort ),
         szSocket, 0 ) == NULL
      )
   {
      String errorMessage = mysql_error( conn );
      errorMessage.bufferize();
      mysql_close( conn );

      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNECT, __LINE__)
              .extra( errorMessage )
           );
   }

   return new DBIHandleMySQL( conn );
}

CoreObject *DBIServiceMySQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findWKI( "MySQL" );
   if ( cl == 0 || ! cl->isClass() )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__ ) );
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );

   return obj;
}

} /* namespace Falcon */

/* end of mysql_srv.cpp */


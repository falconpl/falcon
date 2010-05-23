/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_mod.cpp
 *
 * MySQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 22 May 2010 14:44:49 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <errmsg.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "mysql_mod.h"

namespace Falcon
{

/******************************************************************************
 * Private class used to convert timestamp to MySQL format.
 *****************************************************************************/

class DBITimeConverter_MYSQL_TIME: public DBITimeConverter
{
public:
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
} DBITimeConverter_MYSQL_TIME_impl;

void DBITimeConverter_MYSQL_TIME::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( ((unsigned)bufsize) >= sizeof( MYSQL_TIME ) );

   MYSQL_TIME* mtime = (MYSQL_TIME*) buffer;
   mtime->year = (unsigned) ts->m_year;
   mtime->month = (unsigned) ts->m_month;
   mtime->day = (unsigned) ts->m_day;
   mtime->hour = (unsigned) ts->m_hour;
   mtime->minute = (unsigned) ts->m_minute;
   mtime->second = (unsigned) ts->m_second;
   mtime->second_part = (unsigned) ts->m_msec;
   mtime->neg = 0;

   bufsize = sizeof( MYSQL_TIME );
}


/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

MyDBIInBind::MyDBIInBind():
      m_mybind(0)
{}

MyDBIInBind::~MyDBIInBind()
{
   memFree( m_mybind );
}


void MyDBIInBind::onFirstBinding( int size )
{
   m_mybind = (MYSQL_BIND*) memAlloc( sizeof(MYSQL_BIND) * size );
   memset( m_mybind, 0, sizeof(MYSQL_BIND) * size );
}


void MyDBIInBind::onItemChanged( int num )
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

DBIRecordsetMySQL::DBIRecordsetMySQL( DBIHandleMySQL *dbh, MYSQL_RES *res, MYSQL_STMT *stmt )
    : DBIRecordset( dbh ),
      m_res( res ),
      m_stmt( stmt )
{
   m_row = -1; // BOF
   m_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
   m_columnCount = mysql_num_fields( res );
   m_fields = mysql_fetch_fields( res );

   // bind the output values
   m_pMyBind = (MYSQL_BIND*) memAlloc( sizeof( MYSQL_BIND ) * m_columnCount );
   memset( m_pMyBind, 0, sizeof( MYSQL_BIND ) * m_columnCount );
   m_pOutBind = new MyDBIOutBind[ m_columnCount ];

   // keep track of blobs: they myst be zeroed before each fetch
   m_pBlobId = new int[m_columnCount];
   m_nBlobCount = 0;

   for ( int c = 0; c < m_columnCount; c++ )
   {
      // blob field? -- we need to know its length.
      MYSQL_FIELD& field = m_fields[c];
      MyDBIOutBind& ob = m_pOutBind[c];
      MYSQL_BIND& mb = m_pMyBind[c];

      mb.buffer_type = field.type;
      // Accept to blob in sizes < 1024
      if( field.length >= 1024 && (
            field.type == MYSQL_TYPE_TINY_BLOB ||
            field.type == MYSQL_TYPE_BLOB ||
            field.type == MYSQL_TYPE_MEDIUM_BLOB ||
            field.type == MYSQL_TYPE_LONG_BLOB )
         )
      {
         // if we have a large blob, fetch it separately
         m_pBlobId[m_nBlobCount++] = c;
      }
      else
      {
         mb.buffer_length = field.length + 1;
         mb.buffer = ob.reserve( field.length + 1 );
      }

      mb.length = &ob.nLength;
      mb.is_null = &ob.bIsNull;

   }

   if( mysql_stmt_bind_result( m_stmt, m_pMyBind ) != 0 )
   {
      dbh->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_MIX );
   }

}

DBIRecordsetMySQL::~DBIRecordsetMySQL()
{
   if ( m_res != NULL )
      close();

   memFree( m_pMyBind );
   delete m_pOutBind;
   delete[] m_pBlobId;
}



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
   if ( m_row == -1 || nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   // if the field is nil, return nil
   if( *m_pMyBind[nCol].is_null )
   {
      value.setNil();
      return true;
   }

   unsigned long dlen = *m_pMyBind[nCol].length;
   MyDBIOutBind& outbind = m_pOutBind[nCol];

   switch( m_fields[nCol].type )
   {
   case MYSQL_TYPE_NULL:
      value.setNil();
      break;

   case MYSQL_TYPE_TINY:
     value.setInteger( (*(char*) outbind.memory() ) );

      break;

   case MYSQL_TYPE_YEAR:
   case MYSQL_TYPE_SHORT:
      value.setInteger( (*(short*) outbind.memory() ) );
      break;

   case MYSQL_TYPE_INT24:
   case MYSQL_TYPE_LONG:
   case MYSQL_TYPE_ENUM:
   case MYSQL_TYPE_GEOMETRY:
      value.setInteger( (*(int32*) outbind.memory() ) );
      break;

   case MYSQL_TYPE_LONGLONG:
      value.setInteger( (*(int64*) outbind.memory() ) );
      break;

   case MYSQL_TYPE_FLOAT:
      value.setNumeric( (*(float*) outbind.memory() ) );
      break;

   case MYSQL_TYPE_DOUBLE:
      value.setNumeric( (*(double*) outbind.memory() ) );
      break;

   case MYSQL_TYPE_DECIMAL:
   case MYSQL_TYPE_NEWDECIMAL:
      {
         // encoding is utf-8, and values are in range < 127
         String sv = (char*) outbind.memory();
         double dv=0.0;
         sv.parseDouble(dv);
         value.setNumeric(dv);
      }
      break;

   case MYSQL_TYPE_DATE:
   case MYSQL_TYPE_TIME:
   case MYSQL_TYPE_DATETIME:
   case MYSQL_TYPE_TIMESTAMP:
   case MYSQL_TYPE_NEWDATE:
      {
         VMachine* vm = VMachine::getCurrent();
         if( vm == 0 )
         {
            return false;
         }
         CoreObject *ots = vm->findWKI("TimeStamp")->asClass()->createInstance();
         MYSQL_TIME* mtime = (MYSQL_TIME*) outbind.memory();
         TimeStamp* ts = new TimeStamp;

         ts->m_year = mtime->year;
         ts->m_month = mtime->month;
         ts->m_day = mtime->day;
         ts->m_hour = mtime->hour;
         ts->m_minute = mtime->minute;
         ts->m_second = mtime->second;
         ts->m_msec = mtime->second_part;

         ots->setUserData( ts );
         value = ots;
      }
      break;

   // string types
   case MYSQL_TYPE_BIT:
   case MYSQL_TYPE_STRING:
   case MYSQL_TYPE_VARCHAR:
   case MYSQL_TYPE_VAR_STRING:
      // text?
      if( m_fields[nCol].charsetnr == 63 ) // sic -- from manual
      {
         value = new MemBuf_1( (byte*) outbind.memory(), dlen );
      }
      else
      {
         ((char*) outbind.memory())[ dlen ] = 0;
         CoreString* res = new CoreString;
         res->fromUTF8( (char*) outbind.memory() );
         value = res;
      }
   break;

   case MYSQL_TYPE_TINY_BLOB:
   case MYSQL_TYPE_BLOB:
   case MYSQL_TYPE_MEDIUM_BLOB:
   case MYSQL_TYPE_LONG_BLOB:
      // read the missing memory -- and be sure to alloc
      if( dlen != 0 )
      {
         outbind.alloc( dlen );
         m_pMyBind[nCol].buffer_length = dlen;
         m_pMyBind[nCol].buffer = outbind.memory();
         if(  mysql_stmt_fetch_column( m_stmt, m_pMyBind + nCol, nCol, 0 ) != 0 )
         {
            static_cast< DBIHandleMySQL *>(m_dbh)
                  ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
         }
      }

      // text?
      if( m_fields[nCol].charsetnr == 63 ) // sic -- from manual
      {
         // give ownership
         if( dlen == 0 )
         {
            value = new MemBuf_1( 0, 0 );
         }
         else
         {
            value = new MemBuf_1(
               (byte*) outbind.getMemory(),
               dlen,
               memFree );
         }
      }
      else
      {
         if( dlen == 0 )
         {
            value = new CoreString( "" );
         }
         else
         {
            ((char*) outbind.memory() )[ dlen ] = 0;
            CoreString* res = new CoreString;
            res->fromUTF8( (char*) m_pMyBind[nCol].buffer );
            value = res;
         }
      }
      break;

   default:
      static_cast< DBIHandleMySQL *>(m_dbh)
         ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNHANDLED_TYPE );
   }

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

bool DBIRecordsetMySQL::discard( int64 ncount )
{
   // we have all the records. We may seek
   if( m_dbh->options()->m_nPrefetch == -1 )
   {
      mysql_stmt_data_seek( m_stmt, (uint64) ncount + (m_row == 0 ? 0 : m_row+1) );
   }
   else
   {
      for ( int64 i = 0; i < ncount; ++i )
      {
         int res = mysql_stmt_fetch( m_stmt );
         if( res == MYSQL_NO_DATA )
            return false;
         if( res == 1 )
         {
            static_cast< DBIHandleMySQL *>(m_dbh)
                                    ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNHANDLED_TYPE );
         }
      }
   }

   return true;
}

bool DBIRecordsetMySQL::fetchRow()
{
   // first, zero excessively long blobs.
   for( int i = 0; i < m_nBlobCount; ++i  )
   {
      MYSQL_BIND& bind = m_pMyBind[ m_pBlobId[i] ];
      bind.buffer_length = 0;
      *bind.length = 0;
      bind.buffer = 0;
   }

   // then do the real fetch
   int res = mysql_stmt_fetch( m_stmt );
   if( res == 1 )
   {
      // there's an error.
      static_cast< DBIHandleMySQL *>(m_dbh)
            ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
   }
   else if ( res == MYSQL_NO_DATA )
   {
      return false;
   }

   // we have the values
   m_row++;
   return true;
}

void DBIRecordsetMySQL::close()
{
   if ( m_res != 0 && m_stmt != 0 ) {
      mysql_free_result( m_res );
      mysql_stmt_close( m_stmt );
      m_res = 0;
      m_stmt = 0;
   }
}


/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBIStatementMySQL::DBIStatementMySQL( DBIHandle *dbh, MYSQL_STMT* stmt ):
      DBIStatement( dbh ),
      m_statement( stmt ),
      m_inBind(0)
{
}


DBIStatementMySQL::~DBIStatementMySQL()
{
   close();
}


int64 DBIStatementMySQL::execute( const ItemArray& params )
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
         m_inBind = new MyDBIInBind;
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
   return mysql_stmt_affected_rows( m_statement );
}


void DBIStatementMySQL::reset()
{
   if ( m_statement == 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNPREP_EXEC );
   }

   if( mysql_stmt_reset( m_statement ) != 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_RESET );
   }
}


void DBIStatementMySQL::close()
{
   if ( m_statement != 0 )
  {
     mysql_stmt_close( m_statement );
     m_statement = 0;
     delete m_inBind;
     m_inBind = 0;
  }
}



/******************************************************************************
 * DB Handler class
 *****************************************************************************/
DBIHandleMySQL::~DBIHandleMySQL()
{
   DBIHandleMySQL::close();
}

void DBIHandleMySQL::options( const String& params )
{
   if( m_settings.parse( params ) )
   {
      mysql_autocommit( m_conn, m_settings.m_bAutocommit ? 1 : 0);
   }
   else
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_OPTPARAMS );
   }
}

const DBISettingParams* DBIHandleMySQL::options() const
{
   return &m_settings;
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

   mysql_autocommit( m_conn, m_settings.m_bAutocommit ? 1 : 0 );
}

DBIRecordset *DBIHandleMySQL::query( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // prepare and execute -- will create a new m_statement
   MYSQL_STMT* stmt = my_prepare( sql );
   MYSQL_RES* meta = 0;

   try
   {
      MyDBIInBind bindings;
      affectedRows = my_execute( stmt, bindings, params );

      // We want a result recordset
      meta = mysql_stmt_result_metadata( stmt );
      if( meta == 0 )
      {
         // the query didn't return a recorset
         throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY_EMPTY );
      }
      else
      {
         // ok. Do the user wanted all the result back?
         if( m_settings.m_nPrefetch < 0 )
         {
            if( mysql_stmt_store_result( stmt ) != 0 )
            {
               throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
            }
         }

         // -- may throw
         DBIRecordsetMySQL* recset = new DBIRecordsetMySQL( this, meta, stmt );
         return recset;
      }

   }
   catch( ... )
   {
      if( meta != 0 )
         mysql_free_result( meta );

      mysql_stmt_close( stmt );
      throw;
   }


   return 0; // to make the compiler happy
}


void DBIHandleMySQL::perform( const String &sql, int64 &affectedRows, const ItemArray& params )
{
   // if we don't have var params, we can use the standard query, after proper escape.
   if ( params.length() == 0 )
   {
      MYSQL *conn = getConn();

      AutoCString asQuery( sql );
      if( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
      {
         throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
      }

      // discard eventual recordsets
      while ( mysql_next_result( conn ) )
      {
         MYSQL_RES* rec = mysql_use_result( conn );
         if( rec != 0 )
         {
            mysql_free_result(rec);
         }
      }
   }
   else
   {
     MyDBIInBind bindings;

     // prepare and execute
     affectedRows = my_execute(
           my_prepare( sql ),
           bindings,
           params );
   }
}


MYSQL_STMT* DBIHandleMySQL::my_prepare( const String &query )
{
   MYSQL_STMT* stmt = mysql_stmt_init( getConn() );
   if( stmt == 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_NOMEM );
   }

   AutoCString cquery( query );
   if( mysql_stmt_prepare( stmt, cquery.c_str(), cquery.length() ) != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
   }

   // prepare the attributes as suggested by our parameters.
   unsigned long setting = m_settings.m_nCursorThreshold == 0  ?
         CURSOR_TYPE_READ_ONLY : CURSOR_TYPE_NO_CURSOR;

   mysql_stmt_attr_set( stmt, STMT_ATTR_CURSOR_TYPE, &setting );

   if( m_settings.m_nPrefetch > 0 )
   {
      setting = (unsigned long) m_settings.m_nPrefetch;
      mysql_stmt_attr_set( stmt, STMT_ATTR_PREFETCH_ROWS, &setting );
   }
   else if ( m_settings.m_nPrefetch == -1 )
   {
      setting = (unsigned long) 0xFFFFFFFF;
      mysql_stmt_attr_set( stmt, STMT_ATTR_PREFETCH_ROWS, &setting );
   }

   return stmt;
}


int64 DBIHandleMySQL::my_execute( MYSQL_STMT* stmt, MyDBIInBind& bindings, const ItemArray& params )
{
   if( params.length() != mysql_stmt_param_count( stmt ) )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
   }

   // Do we have some parameter to bind?
   if( params.length() > 0 )
   {
      bindings.bind( params );

      if( mysql_stmt_bind_param( stmt, bindings.mybindings() ) != 0 )
      {
         throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_MIX );
      }
   }

   if( mysql_stmt_execute( stmt ) != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_EXEC );
   }

   // row count?
   return mysql_stmt_affected_rows( stmt );
}


DBIStatement* DBIHandleMySQL::prepare( const String &query )
{
   MYSQL_STMT* stmt = my_prepare( query );
   return new DBIStatementMySQL( this, stmt );
}


int64 DBIHandleMySQL::getLastInsertedId( const String& sequenceName )
{
   return mysql_insert_id( m_conn );
}


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

   if ( errorMessage != NULL )
   {
      String description;
      description.N( (int64) mysql_errno( m_conn ) ).A(": ");
      description.A( errorMessage );
      dbi_throwError( file, line, code, description );
   }
   else
   {
      dbi_throwError( file, line, code, "" );
   }
}

String DBIHandleMySQL::callSP( const String& s ) const
{
   return "CALL " + s;
}


/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceMySQL::init()
{
}

DBIHandle *DBIServiceMySQL::connect( const String &parameters, bool persistent )
{
   MYSQL *conn = mysql_init( NULL );

   if ( conn == NULL )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
   }

   // Parse the connection string.
   DBIConnParams connParams;

   // add MySQL specific parameters
   String sSocket, sFlags;
   const char *szSocket = 0;
   connParams.addParameter( "socket", sSocket, &szSocket );
   connParams.addParameter( "flags", sFlags );

   if( ! connParams.parse( parameters ) )
   {
      mysql_close( conn );
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
         .extra( parameters )
      );
   }

   long szFlags = CLIENT_MULTI_STATEMENTS;
   // TODO parse flags

   if ( mysql_real_connect( conn,
         connParams.m_szHost,
         connParams.m_szUser,
         connParams.m_szPassword,
         connParams.m_szDb,
         connParams.m_szPort == 0 ? 0 : atoi( connParams.m_szPort ),
         szSocket, szFlags ) == NULL
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


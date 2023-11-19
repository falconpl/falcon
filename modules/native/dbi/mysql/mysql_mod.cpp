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
#include <mysqld_error.h>


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

MyDBIInBind::MyDBIInBind( MYSQL_STMT* stmt ):
      m_mybind(0),
      m_stmt( stmt )
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
      *((bool*) item.data()) = 1;
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
      // Blobs in prepared statements are treated strangely:
      // The first push is ok, but the other require either rebind or
      // send_long_data to be called on the statement.
      // It seems that stmt_param_bind calls send_long_data on its own to
      // circumvent a bug -- so, the next time YOU must call it in its place.
      if ( myitem.buffer != 0 )
      {
         mysql_stmt_send_long_data( m_stmt, num, (const char*) item.asBuffer(), item.asStringLen() );
      }
      myitem.buffer = item.asBuffer();
      myitem.buffer_length = item.asStringLen();
      break;

   case DBIBindItem::t_time:
      myitem.buffer_type = MYSQL_TYPE_TIMESTAMP;
      myitem.buffer = item.asBuffer();
      myitem.buffer_length = sizeof( MYSQL_TIME );
      break;
   }
}

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetMySQL::DBIRecordsetMySQL( DBIHandleMySQL *dbh, MYSQL_RES *res, bool bCanSeek )
    : DBIRecordset( dbh ),
      m_res( res ),
      m_bCanSeek( bCanSeek )
{
   m_row = -1; // BOF
   m_rowCount = -1; // default -- not known
   m_columnCount = mysql_num_fields( res );
   m_fields = mysql_fetch_fields( res );
   m_pConn = dbh->getConn();
   m_pConn->incref();
}

DBIRecordsetMySQL::~DBIRecordsetMySQL()
{
   if ( m_res != 0 )
      close();
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
   if ( m_res != 0 ) {
      mysql_free_result( m_res );
      m_res = 0;
      m_pConn->decref();
   }
}

/******************************************************************************
 * Recordset class --- when using statements.
 *****************************************************************************/

DBIRecordsetMySQL_STMT::DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbh, MYSQL_RES *res, MYSQLStmtHandle *pStmt, bool bCanSeek )
    : DBIRecordsetMySQL( dbh, res, bCanSeek ),
      m_stmt( pStmt->handle() ),
      m_pStmt( pStmt )
{
   pStmt->incref();
}

DBIRecordsetMySQL_STMT::DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbh, MYSQL_RES *res, MYSQL_STMT *stmt, bool bCanSeek )
    : DBIRecordsetMySQL( dbh, res, bCanSeek ),
      m_stmt( stmt ),
      m_pStmt( new MYSQLStmtHandle(stmt) )
{
}

void DBIRecordsetMySQL_STMT::init()
{
   // bind the output values
   m_pMyBind = (MYSQL_BIND*) memAlloc( sizeof( MYSQL_BIND ) * m_columnCount );
   memset( m_pMyBind, 0, sizeof( MYSQL_BIND ) * m_columnCount );
   m_pOutBind = new MyDBIOutBind[ m_columnCount ];

   // keep track of blobs: they must be zeroed before each fetch
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
      if( field.type == MYSQL_TYPE_DATE ||
          field.type == MYSQL_TYPE_TIME ||
          field.type == MYSQL_TYPE_DATETIME ||
          field.type == MYSQL_TYPE_TIMESTAMP ||
          field.type == MYSQL_TYPE_NEWDATE
      )
      {
         mb.buffer_length = sizeof( MYSQL_TIME );
         mb.buffer = ob.reserve( mb.buffer_length );
      }
      else if( field.length >= 1024 && (
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
      static_cast<DBIHandleMySQL*>(m_dbh)->
               throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_MIX );
   }

   m_rowCount = mysql_stmt_affected_rows( m_stmt );
}

DBIRecordsetMySQL_STMT::~DBIRecordsetMySQL_STMT()
{
   close();

   memFree( m_pMyBind );
   delete m_pOutBind;
   delete[] m_pBlobId;
}


bool DBIRecordsetMySQL_STMT::getColumnValue( int nCol, Item& value )
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
         MYSQL_TIME* mtime = (MYSQL_TIME*) outbind.memory();
         TimeStamp* ts = new TimeStamp;

         ts->m_year = mtime->year;
         ts->m_month = mtime->month;
         ts->m_day = mtime->day;
         ts->m_hour = mtime->hour;
         ts->m_minute = mtime->minute;
         ts->m_second = mtime->second;
         ts->m_msec = mtime->second_part;

         CoreObject *ots = vm->findWKI("TimeStamp")->asClass()->createInstance();
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
         CoreString* res = new CoreString;
         if( dlen > 0 )
         {
            //((char*) outbind.memory())[ dlen -1] = 0;
            res->fromUTF8( (char*) outbind.memory() );
         }
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
         outbind.alloc( dlen + 1 );
         m_pMyBind[nCol].buffer_length = dlen+1;
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
            ((char*) outbind.memory())[ dlen ] = 0;
            CoreString* res = new CoreString;
            res->fromUTF8( (char*) outbind.memory() );
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

bool DBIRecordsetMySQL_STMT::discard( int64 ncount )
{
   if( m_res == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   // we have all the records. We may seek
   if( m_bCanSeek )
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

bool DBIRecordsetMySQL_STMT::fetchRow()
{
   if( m_res == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

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

void DBIRecordsetMySQL_STMT::close()
{
   if ( m_stmt != 0 && m_pConn != 0)
   {
	   int status = 0;
	#if MYSQL_VERSION_ID > 50500
      do {
         mysql_stmt_free_result(m_stmt);
      }
      while( (status = mysql_stmt_next_result( m_stmt )) == 0 );

      if( status != -1 )
      {
         const char* text = mysql_stmt_error(m_stmt);
         static_cast< DBIHandleMySQL *>(m_dbh)
                ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_CLOSING );
      }
      // must be closed in the statement handler
      status = mysql_stmt_close(m_stmt);
      #else

      while( mysql_next_result( m_pConn->handle() ) == 0 )
      {
         MYSQL_RES *res = mysql_use_result( m_pConn->handle() );
         if( res != NULL )
         {
            mysql_free_result( res );
         }
      }
      #endif

      m_stmt = 0;
      if( status != 0 )
      {
         const char* text = mysql_error(m_pConn->handle());
         static_cast< DBIHandleMySQL *>(m_dbh)
                ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_CLOSING );
      }
   }

   DBIRecordsetMySQL::close();
}

DBIRecordset* DBIRecordsetMySQL_STMT::getNext()
{
   DBIHandleMySQL* mysql = static_cast<DBIHandleMySQL*>(m_dbh);

   if ( mysql_next_result( m_pConn->handle() ) == 0 )
   {
      // We want a result recordset
      MYSQL_RES * meta = mysql_stmt_result_metadata( m_pStmt->handle() );
      if( meta == 0 )
      {
         //No, we have nothing to return.
         mysql->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
      }

      // ok. Do the user wanted all the result back?
      if( m_dbh->options()->m_nPrefetch < 0 )
      {
         if( mysql_stmt_store_result( m_pStmt->handle() ) != 0
             && mysql_errno( m_pConn->handle() ) != 0 )
         {
            mysql_free_result( meta );
            mysql->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
         }
      }

      DBIRecordsetMySQL_STMT* recset = new DBIRecordsetMySQL_STMT( mysql, meta, m_pStmt );

      // -- may throw
      try {
         recset->init();
      }
      catch( ... )
      {
         delete recset;
         throw;
      }

      return recset;
   }

   return 0;
}

/******************************************************************************
 * Recordset class --- when using query.
 *****************************************************************************/

DBIRecordsetMySQL_RES::DBIRecordsetMySQL_RES( DBIHandleMySQL *dbh, MYSQL_RES *res, bool bCanSeek )
    : DBIRecordsetMySQL( dbh, res, bCanSeek )
{
   m_rowCount = mysql_num_rows( res ); // Only valid when using mysql_store_result instead of use_result
}

DBIRecordsetMySQL_RES::~DBIRecordsetMySQL_RES()
{
}


bool DBIRecordsetMySQL_RES::getColumnValue( int nCol, Item& value )
{
   if ( m_row == -1 || nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }


   const char* data = m_rowData[nCol];
   if( data == 0 )
   {
      value.setNil();
      return true;
   }

   switch( m_fields[nCol].type )
   {
   case MYSQL_TYPE_NULL:
      value.setNil();
      break;

   case MYSQL_TYPE_TINY:
   case MYSQL_TYPE_YEAR:
   case MYSQL_TYPE_SHORT:
   case MYSQL_TYPE_INT24:
   case MYSQL_TYPE_LONG:
   case MYSQL_TYPE_ENUM:
   case MYSQL_TYPE_GEOMETRY:
   case MYSQL_TYPE_LONGLONG:
      {
         int64 vn;
         String sv(data);
         sv.parseInt(vn);
         value = vn;
      }
      break;

   case MYSQL_TYPE_FLOAT:
   case MYSQL_TYPE_DOUBLE:
   case MYSQL_TYPE_DECIMAL:
   case MYSQL_TYPE_NEWDECIMAL:
      {
         double vn;
         String sv(data);
         sv.parseDouble(vn);
         value = vn;
      }
      break;

   case MYSQL_TYPE_DATE:
      value = makeTimestamp( String(data) + " 00:00:00");
      break;

   case MYSQL_TYPE_TIME:
      value = makeTimestamp( String( "0000-00-00 " ) + String(data) );
      break;

   case MYSQL_TYPE_DATETIME:
   case MYSQL_TYPE_TIMESTAMP:
   case MYSQL_TYPE_NEWDATE:
      value = makeTimestamp( String(data) );
      break;

   // string types
   case MYSQL_TYPE_BIT:
   case MYSQL_TYPE_STRING:
   case MYSQL_TYPE_VARCHAR:
   case MYSQL_TYPE_VAR_STRING:
   case MYSQL_TYPE_TINY_BLOB:
   case MYSQL_TYPE_BLOB:
   case MYSQL_TYPE_MEDIUM_BLOB:
   case MYSQL_TYPE_LONG_BLOB:      // text?
      if( m_fields[nCol].flags & BINARY_FLAG ) // sic -- from manual
      {
         unsigned long* lengths = mysql_fetch_lengths( m_res );
         byte* mem = (byte*) memAlloc( lengths[nCol] );
         memcpy( mem, data, lengths[nCol] );
         value = new MemBuf_1( mem, lengths[nCol], memFree );
      }
      else
      {
         CoreString* vs = new CoreString;
         vs->fromUTF8( data );
         value = vs;
      }
      break;


   default:
      static_cast< DBIHandleMySQL *>(m_dbh)
         ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNHANDLED_TYPE );
   }

   return true;
}


CoreObject* DBIRecordsetMySQL_RES::makeTimestamp( const String& str )
{
   VMachine* vm = VMachine::getCurrent();
   if( vm == 0 )
   {
      static_cast< DBIHandleMySQL *>(m_dbh)
            ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNHANDLED_TYPE );
   }
   CoreObject *ots = vm->findWKI("TimeStamp")->asClass()->createInstance();
   TimeStamp* ts = new TimeStamp;

   int64 ival;
   str.subString(0,4).parseInt(ival);
   ts->m_year = ival;
   str.subString(5,7).parseInt(ival);
   ts->m_month = ival;
   str.subString(8,10).parseInt(ival);
   ts->m_day = ival;
   str.subString(11,13).parseInt(ival);
   ts->m_hour = ival;
   str.subString(14,16).parseInt(ival);
   ts->m_minute = ival;
   str.subString(17).parseInt(ival);
   ts->m_second = ival;
   ts->m_msec = 0;

   ots->setUserData( ts );
   return ots;
}


bool DBIRecordsetMySQL_RES::discard( int64 ncount )
{
   if( m_res == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   // we have all the records. We may seek
   if( m_dbh->options()->m_nPrefetch == -1 )
   {
      m_row = ncount + (m_row == 0 ? 0 : m_row+1);
      mysql_data_seek( m_res, (uint64) m_row );

   }
   else
   {
      for ( int64 i = 0; i < ncount; ++i )
      {
         MYSQL_ROW row = mysql_fetch_row( m_res );

         if( row == 0 )
         {
            if ( mysql_errno( m_pConn->handle() ) != 0 )
            {
               static_cast< DBIHandleMySQL *>(m_dbh)
                   ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_UNHANDLED_TYPE );
            }
            return false;
         }

         m_row++;
      }
   }

   return true;
}

bool DBIRecordsetMySQL_RES::fetchRow()
{
   if( m_res == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_RSET, __LINE__ ) );

   m_rowData = mysql_fetch_row( m_res );
   if ( m_rowData == 0 )
      return false;

   // we have the values
   m_row++;
   return true;
}


/******************************************************************************
 * Recordset class --- when using query + direct string output.
 *****************************************************************************/

DBIRecordsetMySQL_RES_STR::DBIRecordsetMySQL_RES_STR( DBIHandleMySQL *dbh, MYSQL_RES *res, bool bCanSeek )
    : DBIRecordsetMySQL_RES( dbh, res, bCanSeek )
{
}

DBIRecordsetMySQL_RES_STR::~DBIRecordsetMySQL_RES_STR()
{
	DBIRecordsetMySQL_RES_STR::close();
}


void DBIRecordsetMySQL_RES_STR::close()
{
   if ( m_res != 0 ) {
      int result;
      while( (result = mysql_next_result( m_pConn->handle() )) == 0)
      {
         m_res = mysql_use_result( m_pConn->handle() );
         mysql_free_result( m_res );
      }

      m_res = 0;
      if( result != -1 &&  mysql_errno( m_pConn->handle() ) != 0 ) {
    	  static_cast< DBIHandleMySQL *>(m_dbh)
                 ->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_CLOSING );
      }
      m_pConn->decref();
   }
}


bool DBIRecordsetMySQL_RES_STR::getColumnValue( int nCol, Item& value )
{
   if ( m_row == -1 || nCol < 0 || nCol >= m_columnCount )
   {
      return false;
   }

   const char* data = m_rowData[nCol];
   if( data == 0 || m_fields[nCol].type == MYSQL_TYPE_NULL )
   {
      value.setNil();
   }
   else if( m_fields[nCol].charsetnr == 63 && IS_LONGDATA(m_fields[nCol].type ) ) // sic -- from manual
   {
      unsigned long* lengths = mysql_fetch_lengths( m_res );
      byte* mem = (byte*) memAlloc( lengths[nCol] );
      memcpy( mem, data, lengths[nCol] );
      value = new MemBuf_1( mem, lengths[nCol], memFree );
   }
   else
   {
      CoreString* vs = new CoreString;
      vs->fromUTF8( data );
      value = vs;
   }

   return true;
}


/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBIStatementMySQL::DBIStatementMySQL( DBIHandleMySQL *dbh, MYSQL_STMT* stmt ):
      DBIStatement( dbh ),
      m_statement( stmt ),
      m_inBind(0),
      m_bBound( false )
{
   m_pConn = dbh->getConn();
   m_pConn->incref();
   m_pStmt = new MYSQLStmtHandle( stmt );
}


DBIStatementMySQL::~DBIStatementMySQL()
{
   close();
}


DBIRecordset* DBIStatementMySQL::execute( ItemArray* params )
{
   if( m_statement == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   // should we bind with the statement? -- first time around?
   if ( ! m_bBound )
   {
      m_bBound = true;

      // Do we have some parameter to bind?
      if( params == 0 )
      {
         if( mysql_stmt_param_count( m_statement ) != 0 )
         {
            getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
         }
      }
      else
      {
         if( params->length() != mysql_stmt_param_count( m_statement ) )
         {
            getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
         }

         // params.lengh() == 0 is possible with totally static selects,
         // or other statements that will be run usually just once.
         // Inserts or other repetitive statements will have at least 1, so
         // this branch won't be repeatedly checked in the fast path.
         m_inBind = new MyDBIInBind( m_statement );
         m_inBind->bind( *params, DBITimeConverter_MYSQL_TIME_impl );

         if( mysql_stmt_bind_param( m_statement, m_inBind->mybindings() ) != 0 )
         {
            getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_MIX );
         }
      }
   }
   else
   {
      if ( params != 0 && m_inBind != 0 )
      {
         m_inBind->bind( *params, DBITimeConverter_MYSQL_TIME_impl );
      }
      else if ( m_inBind != 0 )
      {
         // we had parameters, but now we have not.
         getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
      }
   }

   if( mysql_stmt_execute( m_statement ) != 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_EXEC );
   }

   // row count?
   m_nLastAffected = mysql_stmt_affected_rows( m_statement );

   // do we have metadata?
   MYSQL_RES* meta = 0;

   DBIHandleMySQL* mysql = static_cast<DBIHandleMySQL* >(m_dbh);

   // We want a result recordset
   meta = mysql_stmt_result_metadata( m_statement );
   if( meta == 0 )
   {
      //No, we have nothing to return.
      return 0;
   }
   else
   {
      // ok. Do the user wanted all the result back?
      if( mysql->options()->m_nPrefetch < 0 )
      {
         if( mysql_stmt_store_result( m_statement ) != 0 )
         {
            mysql_free_result( meta );
            mysql->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
         }
      }

      DBIRecordsetMySQL_STMT* recset = new DBIRecordsetMySQL_STMT( mysql, meta, m_pStmt );

      // -- may throw
      try {
         recset->init();
      }
      catch( ... )
      {
         delete recset;
         throw;
      }

      return recset;
   }
}


void DBIStatementMySQL::reset()
{
   if( m_statement == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

   if( mysql_stmt_reset( m_statement ) != 0 )
   {
      getMySql()->throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_RESET );
   }
}


void DBIStatementMySQL::close()
{
  if ( m_statement != 0 )
  {
     m_statement = 0;
     delete m_inBind;
     m_inBind = 0;
     m_pConn->decref();
     m_pStmt->decref();
  }
}

MYSQLStmtHandle::~MYSQLStmtHandle()
{
   mysql_stmt_close( handle() );
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
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
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
   m_pConn = new MYSQLHandle( conn );

   // we'll be using UTF-8 charset
   mysql_set_character_set( m_conn, "utf8" );

   mysql_autocommit( m_conn, m_settings.m_bAutocommit ? 1 : 0 );
}


DBIRecordset *DBIHandleMySQL::query_internal( const String &sql, ItemArray* params )
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   // do we want to fetch strings?
   if( ! options()->m_bFetchStrings )
   {
      // if not, try to prepare and execute.
      // prepare and execute -- will create a new m_statement
      MYSQL_STMT* stmt = my_prepare( sql, true );

      // If 0, it means that mysql doesn't support prepare for this query.
      if ( stmt != 0 )
      {
         MYSQL_RES* meta = 0;
         DBIRecordsetMySQL_STMT* recset = 0;

         try
         {
            MyDBIInBind bindings(stmt);
            m_nLastAffected = my_execute( stmt, bindings, params );

            // We want a result recordset
            meta = mysql_stmt_result_metadata( stmt );
            if( meta == 0 )
            {
			#if MYSQL_VERSION_ID > 50500
			   do {
				  mysql_stmt_free_result( stmt );
			   }
			   while( mysql_stmt_next_result(stmt) == 0);
	  	    #endif
			   mysql_stmt_close( stmt );
               return 0;
            }

            // ok. Do the user wanted all the result back?
            if( m_settings.m_nPrefetch < 0 )
            {
               if( mysql_stmt_store_result( stmt ) != 0 )
               {
                  throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_FETCH );
               }
            }

            // -- may throw
            recset = new DBIRecordsetMySQL_STMT( this, meta, stmt );
            recset->init();
            return recset;
         }
         catch( ... )
         {
            if( meta != 0 )
               mysql_free_result( meta );

            if( recset )
            {
               delete recset;
            }
            else
            {
			#if MYSQL_VERSION_ID > 50500
			   do {
				  mysql_stmt_free_result( stmt );
			   }
			   while( mysql_stmt_next_result(stmt) == 0);
	  	    #endif
               mysql_stmt_close( stmt );
            }
            throw;
         }
      }
   }

   // either we WANT to fetch strings, or we're FORCED by mysql
   // -- which may not support prepare/execute for this query.
   MYSQL *conn = m_conn;
   int result;
   if( params != 0)
   {
      String temp;
      sqlExpand( sql, temp, *params );
      AutoCString asQuery( temp );
      result =  mysql_real_query( conn, asQuery.c_str(), asQuery.length() );
   }
   else
   {
      AutoCString asQuery( sql );
      result =  mysql_real_query( conn, asQuery.c_str(), asQuery.length() );
   }

   if( result != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
   }

   MYSQL_RES* rec =  options()->m_nPrefetch < 0 ?
      mysql_store_result( conn ) :
      mysql_use_result( conn );

   m_nLastAffected = mysql_affected_rows( conn );
   if( rec ==  0 )
   {
      return 0;
   }

   return new DBIRecordsetMySQL_RES_STR( this, rec );
}



DBIRecordset *DBIHandleMySQL::query( const String &sql, ItemArray* params )
{
   return query_internal( sql, params );
}

void DBIHandleMySQL::result( const String &sql, Item& res, ItemArray* params )
{
   DBIRecordset* rs = query_internal( sql, params );
   std_result( rs, res );
}


MYSQL_STMT* DBIHandleMySQL::my_prepare( const String &query, bool bCanFallback )
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   MYSQL_STMT* stmt = mysql_stmt_init( m_conn );
   if( stmt == 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_NOMEM );
   }

   AutoCString cquery( query );

   if( mysql_stmt_prepare( stmt, cquery.c_str(), cquery.length() ) != 0 )
   {
      int result = mysql_errno( m_conn );
      if ( result == 1295 && bCanFallback )
      {
         // unsupported as prepared query
         return 0;
      }

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


int64 DBIHandleMySQL::my_execute( MYSQL_STMT* stmt, MyDBIInBind& bindings, ItemArray* params )
{
   fassert( m_conn != 0 );
   int count = mysql_stmt_param_count( stmt );

   if( params == 0 || params->length() == 0 )
   {
      if( count != 0 )
         throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
   }
   else
   {
      // Do we have some parameter to bind?
      if  ( params->length() != count )
      {
         throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_BIND_SIZE );
      }

      bindings.bind( *params, DBITimeConverter_MYSQL_TIME_impl );

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
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   return mysql_insert_id( m_conn );
}


void DBIHandleMySQL::begin()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( mysql_query( m_conn, "BEGIN" ) != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_TRANSACTION );
   }
}


void DBIHandleMySQL::commit()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( mysql_query( m_conn, "COMMIT" ) != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_TRANSACTION );
   }
}


void DBIHandleMySQL::rollback()
{
   if( m_conn == 0 )
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );

   if( mysql_query( m_conn, "ROLLBACK" ) != 0 )
   {
      throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_TRANSACTION );
   }
}


void DBIHandleMySQL::selectLimited( const String& query,
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


void DBIHandleMySQL::close()
{
   if ( m_conn != NULL )
   {
      mysql_query( m_conn, "COMMIT" );
      m_pConn->decref();
      m_conn = NULL;
   }
}

void DBIHandleMySQL::throwError( const char* file, int line, int code )
{
   if( m_conn == 0 ){
	   return;
   }

   const char *errorMessage = mysql_error( m_conn );

   if ( errorMessage != NULL )
   {
      String description;
      description.N( (int64) mysql_errno( m_conn ) ).A(": ");
      description.A( errorMessage );
      throw new DBIError( ErrorParam( code, line )
            .extra(description)
            .module( file ) );
   }
   else
   {
      throw new DBIError( ErrorParam( code, line )
                  .module( file ) );
   }
}


/******************************************************************************
 * Main service class
 *****************************************************************************/

void DBIServiceMySQL::init()
{
}

DBIHandle *DBIServiceMySQL::connect( const String &parameters )
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

   long szFlags = CLIENT_MULTI_STATEMENTS|CLIENT_MULTI_RESULTS;
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
      int en = mysql_errno( conn ) == ER_BAD_DB_ERROR ?
               FALCON_DBI_ERROR_DB_NOTFOUND : FALCON_DBI_ERROR_CONNECT;

      String errorMessage = mysql_error( conn );
      errorMessage.bufferize();
      mysql_close( conn );

      throw new DBIError( ErrorParam( en, __LINE__).extra( errorMessage ) );
   }

   if( connParams.m_sCreate == "always" )
   {
      String sDrop = "drop database IF EXIST " + connParams.m_sDb ;

      AutoCString asQuery( sDrop );
      if( mysql_real_query( conn, asQuery.c_str(), asQuery.length() ) != 0 )
      {
         throw new DBIError( ErrorParam(  FALCON_DBI_ERROR_CONNECT_CREATE, __LINE__ ));
      }

      String sCreate = "create database " + connParams.m_sDb ;
      AutoCString asQuery2( sCreate );
      if( mysql_real_query( conn, asQuery2.c_str(), asQuery2.length() ) != 0 )
      {
         throw new DBIError( ErrorParam(  FALCON_DBI_ERROR_CONNECT_CREATE, __LINE__ ));
      }

   }
   else if ( connParams.m_sCreate == "cond" )
   {
      String sCreate = "create database if not exist " + connParams.m_sDb;
      AutoCString asQuery2( sCreate );
      if( mysql_real_query( conn, asQuery2.c_str(), asQuery2.length() ) != 0 )
      {
         throw new DBIError( ErrorParam(  FALCON_DBI_ERROR_CONNECT_CREATE, __LINE__ ));
      }
   }
   else if( connParams.m_sCreate != "" )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
              .extra( parameters )
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

/* end of mysql_mod.cpp */


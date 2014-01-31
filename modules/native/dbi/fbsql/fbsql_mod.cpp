/*
   FALCON - The Falcon Programming Language.
   FILE: fbsql_mod.cpp

   Firebird Falcon service/driver
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 20 Sep 2010 21:15:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/dbi/fgsql/fgsql_mod.cpp"

#include <string.h>
#include <stdio.h>

#include <time.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include <falcon/stdhandlers.h>
#include <falcon/timestamp.h>
#include <falcon/itemarray.h>
#include "fbsql_mod.h"

namespace Falcon
{

/******************************************************************************
 * Private class used to convert timestamp to Firebird format.
 *****************************************************************************/

class DBITimeConverter_Firebird_TIME: public DBITimeConverter
{
public:
   virtual ~DBITimeConverter_Firebird_TIME() {}
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
} DBITimeConverter_Firebird_TIME_impl;

void DBITimeConverter_Firebird_TIME::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( ((unsigned)bufsize) >= sizeof( ISC_TIMESTAMP ) );

   ISC_TIMESTAMP* mtime = (ISC_TIMESTAMP*) buffer;
   struct tm entry_time;

   entry_time.tm_year = ts->year() < 1900 ? 0 : ts->year() - 1900;
   entry_time.tm_mon = ts->month()-1;
   entry_time.tm_mday = (unsigned) ts->day();
   entry_time.tm_hour = (unsigned) ts->hour();
   entry_time.tm_min = (unsigned) ts->minute();
   entry_time.tm_sec = (unsigned) ts->second();

   isc_encode_timestamp( &entry_time, mtime );
   mtime->timestamp_time += ts->msec()*10;
   bufsize = sizeof( ISC_TIMESTAMP );
}


/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

FBInBind::FBInBind( isc_db_handle dbh, isc_tr_handle tr, isc_stmt_handle stmt ):
      m_dbh( dbh ),
      m_tr(tr),
      m_stmt(stmt),
      m_sqlInd(0),
      m_GIDS(0)
{
}

FBInBind::~FBInBind()
{
   if( m_sqlInd != 0 )
      free(m_sqlInd);

   if( m_GIDS != 0 )
      free( m_GIDS );
}


void FBInBind::onFirstBinding( int size )
{
   m_data.describeIn( m_stmt );
   if( size != m_data.varCount() )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_BIND_SIZE, __LINE__, SRC )
            .extra( String("").N(size).A("!=").N(m_data.varCount())) );
   }
   m_sqlInd = (ISC_SHORT*) malloc( size * sizeof(ISC_SHORT) );
}


void FBInBind::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];
   XSQLVAR* var = m_data.var( num );

   // Normally not nil
   var->sqlind = m_sqlInd+num;
   *var->sqlind = 0;

   switch( item.type() )
   {
   // set to null
   case DBIBindItem::t_nil:
      var->sqltype = SQL_LONG+1;
      var->sqldata = item.userbuffer();
      var->sqllen = sizeof( ISC_LONG );
      *var->sqlind = -1;
      break;

   case DBIBindItem::t_bool:
      var->sqltype = SQL_SHORT;
      var->sqldata = (ISC_SCHAR*) item.userbuffer();
      (*(ISC_SHORT*)var->sqldata) = (ISC_SHORT) (item.asInteger() > 0 ? 1 : 0);
      var->sqllen = sizeof( ISC_SHORT );
      break;

   case DBIBindItem::t_int:
      var->sqltype = SQL_INT64;
      var->sqldata = (ISC_SCHAR*) item.asIntegerPtr();
      var->sqllen = sizeof( int64 );
      break;

   case DBIBindItem::t_double:
      var->sqltype = SQL_DOUBLE;
      var->sqldata = (ISC_SCHAR*) item.asDoublePtr();
      var->sqllen = sizeof( double );
      break;

   case DBIBindItem::t_string:
      var->sqltype = SQL_TEXT;
      var->sqldata = (ISC_SCHAR*) item.asString();
      var->sqllen = item.asStringLen();
      break;

   case DBIBindItem::t_buffer:
      {
         // Create the blob data
         if ( m_GIDS == 0 )
         {
            m_GIDS = (ISC_QUAD*) malloc( sizeof( ISC_QUAD ) * m_size );
         }
         m_GIDS[num] = createBlob( (byte*)item.asBuffer(), item.asStringLen() );
         var->sqltype = SQL_BLOB;
         var->sqldata = (ISC_SCHAR*)(m_GIDS + num);
         var->sqllen = sizeof( ISC_QUAD );
      }
      break;

   case DBIBindItem::t_time:
      var->sqltype = SQL_TIMESTAMP;
      var->sqldata = (ISC_SCHAR*) item.asBuffer();
      var->sqllen = item.asStringLen();
      break;
   }
}


ISC_QUAD FBInBind::createBlob( byte* data, int64 size )
{
   isc_blob_handle handle;
   handle = 0;
   ISC_QUAD blob_id;

   ISC_STATUS status[20];
   ISC_STATUS res;

   res = isc_create_blob2(
         status,
         &m_dbh,
         &m_tr,
         &handle, /* set by this function to refer to the new Blob */
         &blob_id, /* Blob ID set by this function */
         0, /* Blob Parameter Buffer length = 0; no filter will be used */
         NULL /* NULL Blob Parameter Buffer, since no filter will be used */
      );

   if( res != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_EXEC, status );
   }

   int64 done = 0;
   int len = 4096;
   while ( done < size )
   {
      len = (size-done) < 4096 ? (int)(size-done) : 4096;
      res = isc_put_segment( status, &handle, len, (const ISC_SCHAR*)(data+done) );
      if( res != 0 )
      {
         ISC_STATUS dummy[20];
         isc_cancel_blob( dummy, &handle );
         DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_EXEC, status );
      }

      done += len;
   }

   if( isc_close_blob( status, &handle ) != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_EXEC, status );
   }

   return blob_id;
}

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetFB::DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, isc_stmt_handle stmt, FBSqlData* data ):
   DBIRecordset( dbt ),
   m_nRow(0),
   m_nRowCount(-2),
   m_dbref( dbt->connRef() ),
   m_tref(tref),
   m_sref( new FBStmtRef(stmt) ),
   m_data( data )
{
   m_dbref->incref();
   tref->incref();
}


DBIRecordsetFB::DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, FBStmtRef* sref, FBSqlData* data ):
   DBIRecordset( dbt ),
   m_nRow(0),
   m_nRowCount(-2),
   m_dbref( dbt->connRef() ),
   m_tref(tref),
   m_sref( sref ),
   m_data( data )
{
   m_dbref->incref();
   tref->incref();
   sref->incref();
}


DBIRecordsetFB::~DBIRecordsetFB()
{
   close();
}

int DBIRecordsetFB::getColumnCount()
{
   return m_data->varCount();
}

int64 DBIRecordsetFB::getRowIndex()
{
   return m_nRow;
}

int64 DBIRecordsetFB::getRowCount()
{
   if( m_nRowCount == -2 )
   {
      m_nRowCount = DBIHandleFB::getAffected( m_sref->handle(), FALCON_DBI_ERROR_EXEC );
   }

   return m_nRowCount;
}


bool DBIRecordsetFB::getColumnName( int nCol, String& name )
{
   if( nCol < 0 || nCol >= m_data->varCount() )
   {
      return false;
   }

   XSQLVAR* var = m_data->var(nCol);
   if(var->aliasname != 0 &&  var->aliasname_length !=0 )
   {
      name = String((char*)var->aliasname, var->aliasname_length );
   }
   else if( var->ownname != 0 && var->ownname_length != 0 )
   {
      name = String((char*)var->ownname, var->ownname_length );
   }
   else if( var->relname != 0 && var->relname_length != 0 )
   {
      name = String((char*)var->relname, var->relname_length );
   }
   else
   {
      return false;
   }

   name.bufferize();
   return true;
}


bool DBIRecordsetFB::fetchRow()
{
   // first time?
   if( m_nRow == 0 )
   {
      // generate enough space.
      m_data->allocOutput();
   }

   ISC_STATUS status[20];
   ISC_STATUS res;
   // more data incoming
   res = isc_dsql_fetch( status, &m_sref->handle(), 1, m_data->table() );

   // Done?
   if ( res == 100L )
   {
      return false;
   }
   else if( res != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_FETCH, status );
   }

   // else, we can proceed
   m_nRow++;
   return true;
}


bool DBIRecordsetFB::discard( int64 ncount )
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


void DBIRecordsetFB::close()
{
   if( m_tref != 0 )
   {
      m_sref->decref();
      m_sref = 0;
      m_tref->decref();
      m_tref = 0;
      delete m_data;
      m_data = 0;
      m_dbref->decref();
      m_dbref = 0;
   }
}


bool DBIRecordsetFB::getColumnValue( int nCol, Item& value )
{
   static Class* clsTS = Engine::instance()->stdHandlers()->timestampClass();

   if( nCol < 0 || nCol >= m_data->varCount() )
   {
      return false;
   }

   XSQLVAR* var = m_data->var(nCol);
   if( (var->sqltype & 1) && *var->sqlind )
   {
      // null
      value.setNil();
      return true;
   }

   // not null
   char* buf = (char*) var->sqldata;
   int varType = var->sqltype & ~1;

   switch( varType )
   {
     case SQL_VARYING:
         // pascal strings - a short with a length information followed by the data
         value = FALCON_GC_HANDLE(new String);
         value.asString()->fromUTF8(var->sqldata + sizeof(short));
         break;

     case SQL_INT64:
         if ( var->sqlscale < 0 )
             value = (*(int64*)buf) * pow(10.0, var->sqlscale );
         else
             value = *(int64*)buf;
         break;

     case SQL_LONG:
         if (var->sqllen == sizeof(int32) )
         {
            if ( var->sqlscale < 0 )
                value = (*(int32*)buf) * pow(10.0, var->sqlscale );
            else
                value = *(int32*)buf;
            break;
         }
         else
         {
            value = *(int64*)buf;
         }
         break;

     case SQL_SHORT:
        if ( var->sqlscale < 0 )
            value = (*(int32*)buf) * pow(10.0, var->sqlscale );
        else
            value = *(int32*)buf;
        break;

     case SQL_FLOAT:
         value = (numeric) (*(float*)buf);
         break;

     case SQL_DOUBLE:
         value = (numeric) (*(double*)buf);
         break;

     case SQL_TIMESTAMP:
     case SQL_TYPE_TIME:
     case SQL_TYPE_DATE:
     {
        TimeStamp* ts = new TimeStamp;
        struct tm tstamp;
        if( varType == SQL_TIMESTAMP )
        {
           isc_decode_timestamp( (ISC_TIMESTAMP*) buf, &tstamp );
           ts->year(tstamp.tm_year+1900);
           ts->month(tstamp.tm_mon+1);
           ts->day(tstamp.tm_mday);
           ts->hour(tstamp.tm_hour);
           ts->minute(tstamp.tm_min);
           ts->second(tstamp.tm_sec);
           ts->msec( (((ISC_TIMESTAMP*) buf)->timestamp_time/10) %1000 );
        }
        else if ( varType == SQL_TYPE_TIME )
        {
           isc_decode_sql_time( (ISC_TIME*) buf, &tstamp );
           ts->hour( tstamp.tm_hour );
           ts->minute( tstamp.tm_min );
           ts->second( tstamp.tm_sec );
           ts->msec( ((*(ISC_TIME*) buf)/10) %1000 );
        }
        else // date
        {
           isc_decode_sql_date( (ISC_DATE*) buf, &tstamp );
           ts->year( tstamp.tm_year+1900 );
           ts->month( tstamp.tm_mon+1 );
           ts->day( tstamp.tm_mday );
        }

        value = FALCON_GC_STORE( clsTS, ts );
        }
        break;

     case SQL_TEXT:
         value= FALCON_GC_HANDLE( new String );
         value.asString()->fromUTF8(var->sqldata, var->sqllen );
         break;

     case SQL_BLOB:
         value = FALCON_GC_HANDLE(fetchBlob((ISC_QUAD*)buf));
         break;

     case SQL_ARRAY:
         //row[idx] = d->fetchArray(i, (ISC_QUAD*)buf);
         break;

     default:
        return false;
     }

   return true;
}


String* DBIRecordsetFB::fetchBlob( ISC_QUAD *bId )
{
   ISC_STATUS status[20];
   ISC_STATUS res;
   isc_blob_handle handle = 0;

   res = isc_open_blob2(status,
          &m_dbref->handle(),
          &m_tref->handle(),
          &handle,
          bId, 0, 0);

   if ( res != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_FETCH, status );
   }

   unsigned short len = 0;
   int64 fullSize = 0;

   struct chunk {
      int size;
      struct chunk* next;
      char data[4096];
   };

   struct chunk* current = (struct chunk*) malloc( sizeof(struct chunk));
   struct chunk* first = current;
   while( (res = isc_get_segment(status, &handle, &len, 4096, current->data )) == 0 || status[1] == isc_segment ) {
      fullSize += len;
      current->size = len;

      struct chunk* next = (struct chunk*) malloc( sizeof(struct chunk));
      current->next = next;

      current = next;
      current->next = 0;
      current->size = 0;
   }

   if( res != 0 && res != isc_segstr_eof )
   {
      while( first != 0 )
      {
         current = first->next;
         free( first );
         first = current;
      }

       ISC_STATUS dummy[20];
       isc_close_blob(dummy, &handle);
       DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_FETCH, status );
   }

   // try to close
   if( isc_close_blob(status, &handle) != 0 )
   {
      while( first != 0 )
      {
         current = first->next;
         free( first );
         first = current;
      }

      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_FETCH, status );
   }

   // save the membuffer
   String* mb = new String( fullSize );
   mb->toMemBuf();

   fullSize = 0;
   while( first != 0 )
   {
      memcpy( mb->getRawStorage() + fullSize, first->data, first->size );
      fullSize += first->size;

      current = first->next;
      free( first );
      first = current;
   }
   mb->size(fullSize);

   return mb;
}




/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementFB::DBIStatementFB( DBIHandleFB *dbh, FBTransRef* tref, const isc_stmt_handle& stmt, FBSqlData* dt ):
   DBIStatement( dbh ),
   m_statement(stmt),
   m_pTref( tref ),
   m_pStmt( new FBStmtRef(stmt) ),
   m_outData( dt ),
   m_inBind( 0 )
{
   m_pConn = dbh->connRef();
   m_pConn->incref();
   tref->incref();

   m_bAutoCommit = dbh->options()->m_bAutocommit;
   m_bGetAffected = dbh->options()->m_bGetAffected;
}

DBIStatementFB::~DBIStatementFB()
{
   close();
}


DBIRecordset* DBIStatementFB::execute( ItemArray* params )
{
   ISC_STATUS status[20];

   // initialize bindings the first time.
   if ( m_inBind == 0 )
   {
      m_inBind = new FBInBind( m_pConn->handle(), m_pTref->handle(), m_statement );
   }

   if( params != 0 )
   {
      m_inBind->bind( *params, DBITimeConverter_Firebird_TIME_impl );
   }
   else
   {
      m_inBind->unbind();
   }

   if( isc_dsql_execute( status, &m_pTref->handle(), &m_statement, 1, m_inBind->table() ) != 0 )
   {
      DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_EXEC, status );
   }

   if( m_bGetAffected )
   {
      m_nLastAffected = DBIHandleFB::getAffected( m_statement, FALCON_DBI_ERROR_EXEC );
   }

   if ( m_bAutoCommit )
   {
      m_pTref->commitRetaining();
   }


   // do we have a recordset?
   if( m_outData != 0 )
   {
      // return it
      return new DBIRecordsetFB( static_cast<DBIHandleFB*>(m_dbh), m_pTref, m_pStmt, m_outData );
   }

   // nope. Go on
   return 0;
}

void DBIStatementFB::reset()
{
}

void DBIStatementFB::close()
{
   if( m_pStmt != 0 )
   {
      delete m_inBind;

      m_pStmt->decref();
      m_pStmt = 0;
      m_pTref->decref();
      m_pConn->decref();
   }
}

/******************************************************************************
 * Reaensaction handler
 *****************************************************************************/


FBTransRef::~FBTransRef()
{
    // force closing if still not closed.
    if ( ! m_bClosed )
    {
       static ISC_STATUS status[20];
       isc_commit_transaction( status, &handle() );
    }
}


void FBTransRef::commit()
{
    static ISC_STATUS status[20];
    if( isc_commit_transaction( status, &handle() ) != 0 )
    {
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status );
    }
    m_bClosed = true;
    decref();
}

void FBTransRef::commitRetaining()
{
    static ISC_STATUS status[20];
    if( isc_commit_retaining( status, &handle() ) != 0 )
    {
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status );
    }
}

void FBTransRef::rollback()
{
    static ISC_STATUS status[20];
    if( isc_rollback_transaction( status, &handle() ) != 0 )
    {
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status );
    }
    m_bClosed = true;
    decref();
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/


FBSqlData::FBSqlData():
      m_sqlda(0),
      m_indicators(0),
      m_bOwnBuffers( false )
{
   m_sqlda = (XSQLDA*) malloc( XSQLDA_LENGTH(5) );
   m_sqlda->version = SQLDA_VERSION1;
   m_sqlda->sqln = 5;
   m_sqlda->sqld = 0;
}

FBSqlData::~FBSqlData()
{
   release();
}

void FBSqlData::release()
{
   if( m_sqlda != 0 )
   {
      if( m_bOwnBuffers )
      {
         for( int i = 0; i < varCount(); ++i )
         {
            free(var(i)->sqldata);
         }

         free( m_indicators );
      }
      free(m_sqlda);
      m_sqlda = 0;
      m_bOwnBuffers = false;
   }
}


void FBSqlData::describeIn( isc_stmt_handle stmt )
{
   ISC_STATUS status[20];

   if( isc_dsql_describe_bind( status, &stmt, 1, m_sqlda ) != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_BIND_INTERNAL, status );
   }

   if ( m_sqlda->sqld > m_sqlda->sqln )
   {
      int count = m_sqlda->sqld;
      free( m_sqlda );
      m_sqlda = (XSQLDA*) malloc( XSQLDA_LENGTH(count) );
      m_sqlda->version = SQLDA_VERSION1;
      m_sqlda->sqln = count;
      m_sqlda->sqld = 0;
      isc_dsql_describe_bind( status, &stmt, 1, m_sqlda );
   }
}


void FBSqlData::describeOut( isc_stmt_handle stmt )
{
   ISC_STATUS status[20];

   if( isc_dsql_describe( status, &stmt, 1, m_sqlda ) != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_BIND_INTERNAL, status );
   }

   if ( m_sqlda->sqld > m_sqlda->sqln )
   {
      int count = m_sqlda->sqld;
      free( m_sqlda );
      m_sqlda = (XSQLDA*) malloc( XSQLDA_LENGTH(count) );
      m_sqlda->version = SQLDA_VERSION1;
      m_sqlda->sqln = count;
      m_sqlda->sqld = 0;
      isc_dsql_describe( status, &stmt, 1, m_sqlda );
   }
}


void FBSqlData::allocOutput()
{
   m_bOwnBuffers = true;
   m_indicators = (ISC_SHORT*) malloc( sizeof(ISC_SHORT) * varCount() );

   for( int i = 0; i < varCount(); ++i )
   {
      XSQLVAR* v = var(i);
      v->sqldata = (ISC_SCHAR*) malloc( v->sqllen );
      v->sqlind = m_indicators + i;
      *v->sqlind = 0;
   }
}

/******************************************************************************
 * DB Settings
 *****************************************************************************/


DBISettingParamsFB::DBISettingParamsFB():
      m_bGetAffected( true )
{
   addParameter( "getaffected", m_sGetAffected );
}

DBISettingParamsFB::DBISettingParamsFB( const DBISettingParamsFB& other ):
      DBISettingParams( other ),
      m_bGetAffected( other.m_bGetAffected )
{
}


DBISettingParamsFB::~DBISettingParamsFB()
{
// noop
}


bool DBISettingParamsFB::parse( const String& connStr )
{
   if ( ! DBISettingParams::parse( connStr) )
      return false;

   if( ! checkBoolean( m_sGetAffected, m_bGetAffected ) )
      return false;

   return true;
}


/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleFB::DBIHandleFB(const Class* h):
      DBIHandle(h),
      m_bCommitted( false )
{
   m_pConn = 0;
   m_pTrans = 0;
   m_nLastAffected = -1;
}

DBIHandleFB::DBIHandleFB( const Class* h, const isc_db_handle &conn ):
      DBIHandle(h),
      m_bCommitted( false )
{
   m_pConn = new FBConnRef( conn );
   m_pTrans = 0;
   m_nLastAffected = -1;
}

DBIHandleFB::~DBIHandleFB()
{
   // don't call close -- close will throw in case of commit error.
   if( m_pConn != 0 )
   {
      if ( m_pTrans != 0 )
      {
         //... and don't call commit; decref will commit without error report on destroy
         m_pTrans->decref();
         m_pTrans = 0;
      }

      m_pConn->decref();
      m_pConn = 0;
   }
}


isc_db_handle DBIHandleFB::getConnData()
{
   if( m_pConn == 0 )
   {
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__, SRC ) );
   }

   return m_pConn->handle();
}



static void checkParamNumber(char *&dpb, const String& value, byte dpb_type, const String &option )
{
   if ( value.length() )
   {
      int64 res;
      if( ! value.parseInt(res) )
         throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__, SRC )
                  .extra( option + "=" + value )
               );

      *dpb++ = dpb_type;
      *dpb++ = 1;
      *dpb++ = (byte) res;
   }
}

static void checkParamYesOrNo(char *&dpb, const String& value, byte dpb_type, const String &option )
{
   if ( value.size() )
   {
      *dpb++ = dpb_type;
      *dpb++ = 1;

     if( value.compareIgnoreCase( "yes" ) == 0 )
        *dpb++ = (byte) 1;
     else if( value.compareIgnoreCase( "no" ) == 0 )
        *dpb++ = (byte) 0;
     else
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__, SRC )
                 .extra( option + "=" + value )
              );
   }
}

static void checkParamString(char *&dpb, const String& value, const char* szValue, byte dpb_type )
{
   if ( value.size() )
   {
       *dpb = dpb_type;
       ++dpb;
       *dpb = (char) value.size();
       ++dpb;
       strcpy( dpb, szValue );
       dpb += value.size();
   }
}


void DBIHandleFB::connect( const String &parameters )
{
   isc_db_handle handle = 0L;

   char dpb_buffer[256*10], *dpb;
   // User name (uid)
   // Password (pwd)

   dpb = dpb_buffer;

   // Parse the connection string.
   DBIConnParams connParams;

   // add Firebird specific parameters
   // Encrypted password (epwd)
   String sPwdEnc; const char* szPwdEncode;
   connParams.addParameter( "epwd", sPwdEnc, &szPwdEncode );

   // Role name (role)
   String sRole; const char* szRole;
   connParams.addParameter( "role", sRole, &szRole );

   // System database administrator’s user name (sa)
   String sSAName; const char* szSAName;
   connParams.addParameter( "sa", sSAName, &szSAName );

   // Authorization key for a software license (license)
   String sLicense; const char* szLicense;
   connParams.addParameter( "license", sLicense, &szLicense );

   // Database encryption key (ekey)
   String sKey; const char* szKey;
   connParams.addParameter( "ekey", sKey, &szKey );

   // Number of cache buffers (nbuf)
   String sNBuf;
   connParams.addParameter( "nbuf", sNBuf );

   // dbkey context scope (kscope)
   String sDBKeyScope;
   connParams.addParameter( "kscope", sDBKeyScope );

   // Specify whether or not to reserve a small amount of space on each database
   // --- page for holding backup versions of records when modifications are made (noreserve)
   String sNoRserve;
   connParams.addParameter( "reserve", sNoRserve );

   // Specify whether or not the database should be marked as damaged (dmg)
   String sDmg;
   connParams.addParameter( "dmg", sDmg );

   // Perform consistency checking of internal structures (verify)
   String sVerify;
   connParams.addParameter( "verify", sVerify );

   // Activate the database shadow, an optional, duplicate, in-sync copy of the database (shadow)
   String sShadow;
   connParams.addParameter( "shadow", sShadow );

   // Delete the database shadow (delshadow)
   String sDelShadow;
   connParams.addParameter( "delshadow", sDelShadow );

   // Activate a replay logging system to keep track of all database calls (beginlog)
   String sBeginLog;
   connParams.addParameter( "beginlog", sBeginLog );

   // Deactivate the replay logging system (quitlog)
   String sQuitLog;
   connParams.addParameter( "quitlog", sQuitLog );

   // Language-specific message file  (lcmsg)
   String sLcMsg; const char* szLcMsg;
   connParams.addParameter( "lcmsg", sLcMsg, &szLcMsg );

   // Character set to be utilized (lctype)
   String sLcType; const char* szLcType;
   connParams.addParameter( "lctype", sLcType, &szLcType );

   // Connection timeout (tout)
   String sTimeout;
   connParams.addParameter( "tout", sTimeout );

   if( ! connParams.parse( parameters ) )
   {
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__, SRC )
         .extra( parameters )
      );
   }

   // create the dpb; first the numerical values.
   *dpb++ = isc_dpb_version1;

   checkParamNumber( dpb, sNBuf, isc_dpb_num_buffers, "nbuf" );
   checkParamNumber( dpb, sTimeout, isc_dpb_connect_timeout, "tout" );

   checkParamYesOrNo( dpb, sDBKeyScope, isc_dpb_no_reserve, "kscope" );
   checkParamYesOrNo( dpb, sNoRserve, isc_dpb_no_reserve, "reserve" );
   checkParamYesOrNo( dpb, sDmg, isc_dpb_damaged, "dmg" );
   checkParamYesOrNo( dpb, sVerify, isc_dpb_verify, "verify" );
   checkParamYesOrNo( dpb, sShadow, isc_dpb_activate_shadow, "shadow" );
   checkParamYesOrNo( dpb, sDelShadow, isc_dpb_delete_shadow, "delshadow" );
   checkParamYesOrNo( dpb, sBeginLog, isc_dpb_begin_log, "beginlog" );
   checkParamYesOrNo( dpb, sQuitLog, isc_dpb_quit_log, "sQuitLog" );

   checkParamString( dpb, connParams.m_sUser, connParams.m_szUser, isc_dpb_user_name );
   checkParamString( dpb, connParams.m_sPassword, connParams.m_szPassword, isc_dpb_password );
   checkParamString( dpb, sPwdEnc, szPwdEncode, isc_dpb_password_enc );
   checkParamString( dpb, sRole, szRole, isc_dpb_sql_role_name );
   checkParamString( dpb, sLicense, szLicense, isc_dpb_license );
   checkParamString( dpb, sKey, szKey, isc_dpb_encrypt_key );
   //checkParamString( dpb, sLcMsg, szLcMsg, isc_dpb_lc_messages );
   // We'll ALWAYS use AutoCString to talk with Firebird, as such we'll ALWAYS use UTF8
   checkParamString( dpb, "UTF8", "UTF8", isc_dpb_lc_messages );

   /* Attach to the database. */
   ISC_STATUS status_vector[20];

   isc_attach_database(status_vector, strlen(connParams.m_szDb), connParams.m_szDb, &handle,
         dpb-dpb_buffer,
         dpb_buffer);

   if ( status_vector[0] == 1 && status_vector[1] )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_CONNECT, status_vector );
   }

   m_pConn = new FBConnRef( handle );
}


void DBIHandleFB::options( const String& params )
{
   if( ! m_settings.parse( params ) )
   {
      // autocommit status is read on query
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__, SRC )
            .extra( params ) );
   }
}

const DBISettingParamsFB* DBIHandleFB::options() const
{
   return &m_settings;
}


DBIRecordset *DBIHandleFB::query( const String &sql, ItemArray* params )
{
   m_nLastAffected = -1;

   // may throw
   isc_stmt_handle stmt = internal_prepare( sql );

   // internal_prepare sets up the transaction
   isc_tr_handle tr1 = m_pTrans->handle();

   ISC_STATUS status[20];
   ISC_STATUS res;

   FBSqlData* out_tab = 0;
   // call the query
   try
   {
      if( params == 0 )
      {
         res = isc_dsql_execute( status, &tr1, &stmt, 1, 0 );
      }
      else
      {
         FBInBind bindings( m_pConn->handle(), tr1, stmt );
         bindings.bind( *params, DBITimeConverter_Firebird_TIME_impl );
         res = isc_dsql_execute( status, &tr1, &stmt, 1, bindings.table() );
      }

      if( res != 0 )
      {
         throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status );
      }

      if( options()->m_bAutocommit )
      {
         m_pTrans->commitRetaining();
      }

      // get affected rows
      if ( m_settings.m_bGetAffected )
      {
         m_nLastAffected = getAffected( stmt );
      }

      // TODO
      // use isc_info_sql_num_variables before allocating FBSqlData
      out_tab = new FBSqlData;
      out_tab->describeOut( stmt );
      if( out_tab->varCount() != 0 )
      {
         return new DBIRecordsetFB( this, m_pTrans, stmt, out_tab );
      }
      else
      {
         if( isc_dsql_free_statement( status, &stmt, DSQL_drop ) != 0 )
         {
            throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status );
         }

         delete out_tab;
         return 0;
      }

   }
   catch( ... )
   {
      delete out_tab;
      isc_dsql_free_statement( status, &stmt, DSQL_drop );
      throw;
   }

   // never actually reached
   return 0;
}



isc_stmt_handle DBIHandleFB::internal_prepare( const String& query )
{
   ISC_STATUS status[20];

   isc_db_handle db1 = getConnData();

   // Open a transaction if we don't have any.
   if ( m_pTrans == 0 )
   {
      begin();
   }
   isc_tr_handle tr1 = m_pTrans->handle();

   // Get a new statement handle
   isc_stmt_handle stmt;
   stmt = NULL;
   if ( isc_dsql_allocate_statement(status, &db1, &stmt) != 0 )
   {
      throwError( __LINE__, FALCON_DBI_ERROR_NOMEM, status );
   }

   AutoCString asQuery( query );
   // TODO: can we remove out_table here?
   if( isc_dsql_prepare( status, &tr1, &stmt, asQuery.length(), asQuery.c_str(), 3, 0 ) != 0 )
   {
      ISC_STATUS dummy_status[20];
      isc_dsql_free_statement( dummy_status, &stmt, DSQL_drop );
      throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status );
   }

   return stmt;
}


DBIStatement* DBIHandleFB::prepare( const String &query )
{
   isc_stmt_handle stmt = internal_prepare( query );
   FBSqlData* out_tab = new FBSqlData;
   out_tab->describeOut( stmt );
   if( out_tab->varCount() != 0 )
   {
      return new DBIStatementFB( this, m_pTrans, stmt, out_tab );
   }
   else
   {
      delete out_tab;
      return new DBIStatementFB( this, m_pTrans, stmt, 0 );
   }

}


int64 DBIHandleFB::getLastInsertedId( const String& )
{
   //isc_db_handle handle = getConnData();
   return 0;
   // TODO
}


int64 DBIHandleFB::getAffected( isc_stmt_handle stmt, int etype )
{
   ISC_STATUS status[20];

   // First, determine the statement type
   char acBuffer[9];
   char qType = isc_info_sql_stmt_type;

   if( isc_dsql_sql_info(status, &stmt, 1, &qType, sizeof(acBuffer), acBuffer) != 0 )
   {
      throwError( __LINE__, etype, status );
   }

   int iLength = isc_vax_integer(&acBuffer[1], 2);
   int queryType = (int) isc_vax_integer(&acBuffer[3], iLength);

   // then, get the right info.
   char info = isc_info_sql_records;
   char buffer[64];
   if( isc_dsql_sql_info( status, &stmt, 1, &info, sizeof(buffer), buffer ) != 0 )
   {
      throwError( __LINE__, etype, status );
   }

   int cCountType = 0;

   switch (queryType) {
   case isc_info_sql_stmt_select:
       cCountType = isc_info_req_select_count;
       break;
   case isc_info_sql_stmt_update:
       cCountType = isc_info_req_update_count;
       break;
   case isc_info_sql_stmt_delete:
       cCountType = isc_info_req_delete_count;
       break;
   case isc_info_sql_stmt_insert:
       cCountType = isc_info_req_insert_count;
       break;
   default:
      // not affecting any row
      return 0;
   }

   for (char *pcBuf = buffer + 3; *pcBuf != isc_info_end; /*nothing*/) {
       char cType = *pcBuf++;
       short sLength = isc_vax_integer (pcBuf, 2);
       pcBuf += 2;
       int iValue = isc_vax_integer (pcBuf, sLength);
       pcBuf += sLength;

       if (cType == cCountType) {
           return iValue;
       }
   }

   // unknown
   return -1;
}


void DBIHandleFB::begin()
{
   ISC_STATUS status[20];

   isc_db_handle handle = getConnData();

   // have we an open transaction?
   if ( m_pTrans != 0 )
   {
      // this closes and decrefs
      m_pTrans->commit();
   }

   // Try to open new transaction
   isc_tr_handle htr = 0L;

   // TODO Use options to create different transaction types
   char isc_tbp[] = {isc_tpb_version3,
      isc_tpb_write,
      isc_tpb_concurrency,
      isc_tpb_wait};

   /* Code for attaching to database here is omitted. */
   ISC_STATUS res = isc_start_transaction(status,
      &htr,
      1,
      &handle,
      (unsigned short) sizeof(isc_tbp),
      isc_tbp);

   if( res != 0 )
   {
      throwError( __LINE__, FALCON_DBI_ERROR_TRANSACTION, status );
   }

   m_pTrans = new FBTransRef( htr );
}


void DBIHandleFB::commit()
{
   getConnData();  // check for db open.
   m_pTrans->commitRetaining(); // don't really close the transaction.
}


void DBIHandleFB::rollback()
{
   getConnData();  // check for db open.
   m_pTrans->rollback(); // this closes and decref the trans.
   m_pTrans = 0; // say we need a new transaction
}


void DBIHandleFB::selectLimited( const String& query,
      int64 nBegin, int64 nCount, String& result )
{
   String sSkip, sCount;

   if ( nBegin > 0 )
   {
      sSkip = " SKIP ";
      sSkip.N( nBegin );
   }

   if( nCount > 0 )
   {
      sCount = " FIRST ";
      sCount.N( nCount );
   }

   result = "SELECT" + sCount + sSkip +" "+ query;
}


void DBIHandleFB::close()
{
   if ( m_pTrans != 0 )
   {
      m_pTrans->commit();  // commit decrefs, and eventually throws
      m_pTrans = 0;
   }

   if( m_pConn != 0 )
   {
      m_pConn->decref();
      m_pConn = 0;
   }
}


void DBIHandleFB::throwError( int line, int code, ISC_STATUS* status )
{
   String desc;
   ISC_SCHAR msgBuffer[512];

   // Get the main error
   const ISC_STATUS* ep;
   ep = status;
   fb_interpret( msgBuffer, 512, &ep );
   desc += msgBuffer;

   // Write the secondary errors as a list of [...; ...; ...] messages
   bool bDone = false;
   while( fb_interpret( msgBuffer, 512, &ep ) )
   {
      if ( ! bDone )
      {
         desc += " [";
         bDone = true;
      }
      else {
         desc += "; ";
      }
      desc += msgBuffer;
   }

   if( bDone )
   {
      desc += "]";
   }

   throw new DBIError( ErrorParam( code, line, SRC ).extra(desc) );
}

} /* namespace Falcon */

/* end of fbsql_mod.cpp */


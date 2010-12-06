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

#include <string.h>
#include <stdio.h>

#include <time.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "fbsql_mod.h"

namespace Falcon
{

DBIServiceFB theFirebirdService;

/******************************************************************************
 * Private class used to convert timestamp to Firebird format.
 *****************************************************************************/

class DBITimeConverter_Firebird_TIME: public DBITimeConverter
{
public:
   virtual void convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const;
} DBITimeConverter_Firebird_TIME_impl;

void DBITimeConverter_Firebird_TIME::convertTime( TimeStamp* ts, void* buffer, int& bufsize ) const
{
   fassert( ((unsigned)bufsize) >= sizeof( ISC_TIMESTAMP ) );

   ISC_TIMESTAMP* mtime = (ISC_TIMESTAMP*) buffer;
   struct tm entry_time;

   entry_time.tm_year = ts->m_year < 1900 ? 0 : ts->m_year - 1900;
   entry_time.tm_mon = ts->m_month-1;
   entry_time.tm_mday = (unsigned) ts->m_day;
   entry_time.tm_hour = (unsigned) ts->m_hour;
   entry_time.tm_min = (unsigned) ts->m_minute;
   entry_time.tm_sec = (unsigned) ts->m_second;

   isc_encode_timestamp( &entry_time, mtime );
   bufsize = sizeof( ISC_TIMESTAMP );
}


/******************************************************************************
 * (Input) bindings class
 *****************************************************************************/

FBInBind::FBInBind( isc_stmt_handle stmt ):
      m_stmt(stmt)
{}

FBInBind::~FBInBind()
{
}


void FBInBind::onFirstBinding( int size )
{
}


void FBInBind::onItemChanged( int num )
{
   DBIBindItem& item = m_ibind[num];
}


/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordsetFB::DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, isc_stmt_handle stmt, FBSqlData* data ):
   DBIRecordset( dbt ),
   m_tref(tref),
   m_sref( new FBStmtRef(stmt) ),
   m_data( data )
{
   tref->incref();
}


DBIRecordsetFB::DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, FBStmtRef* sref, FBSqlData* data ):
   DBIRecordset( dbt ),
   m_tref(tref),
   m_sref( sref ),
   m_data( data )
{
   tref->incref();
   sref->incref();
}


DBIRecordsetFB::~DBIRecordsetFB()
{
   close();
}

int DBIRecordsetFB::getColumnCount()
{
   return m_nColumnCount;
}

int64 DBIRecordsetFB::getRowIndex()
{
   return m_nRow;
}

int64 DBIRecordsetFB::getRowCount()
{
   return m_nRowCount;
}


bool DBIRecordsetFB::getColumnName( int nCol, String& name )
{
}


bool DBIRecordsetFB::fetchRow()
{
   // more data incoming
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
      m_tref->decref();
      m_tref = 0;
      m_sref->decref();
      m_sref = 0;
      delete m_data;
      m_data = 0;
   }
}


bool DBIRecordsetFB::getColumnValue( int nCol, Item& value )
{

   return true;
}


/******************************************************************************
 * DB Statement class
 *****************************************************************************/

DBIStatementFB::DBIStatementFB( DBIHandleFB *dbh, const isc_stmt_handle& stmt ):
   DBIStatement( dbh ),
   m_statement(stmt),
   m_pStmt( new FBStmtRef(stmt) )
{
   m_pConn = dbh->connRef();
   m_pConn->incref();
}

DBIStatementFB::~DBIStatementFB()
{
   close();
}


DBIRecordset* DBIStatementFB::execute( ItemArray* params )
{
   return 0;
}

void DBIStatementFB::reset()
{
}

void DBIStatementFB::close()
{
   if( m_pStmt != 0 )
   {
      m_pStmt->decref();
      m_pStmt = 0;
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
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status, false );
    }
    m_bClosed = true;
    decref();
}

void FBTransRef::commitRetaining()
{
    static ISC_STATUS status[20];
    if( isc_commit_retaining( status, &handle() ) != 0 )
    {
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status, false );
    }
}

void FBTransRef::rollback()
{
    static ISC_STATUS status[20];
    if( isc_rollback_transaction( status, &handle() ) != 0 )
    {
       DBIHandleFB::throwError(__LINE__, FALCON_DBI_ERROR_TRANSACTION, status, false );
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
   m_sqlda = (XSQLDA*) memAlloc( XSQLDA_LENGTH(5) );
   m_sqlda->version = SQLDA_VERSION1;
   m_sqlda->sqln = 5;
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
            memFree(var(i)->sqldata);
         }

         memFree( m_indicators );
      }
      memFree(m_sqlda);
      m_sqlda = 0;
      m_bOwnBuffers = false;
   }
}


void FBSqlData::describeIn( isc_stmt_handle stmt )
{
   ISC_STATUS status[20];

   if( isc_dsql_describe_bind( status, &stmt, 1, m_sqlda ) != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_BIND_INTERNAL, status, true );
   }

   if ( m_sqlda->sqld > m_sqlda->sqln )
   {
      int count = m_sqlda->sqld;
      memFree( m_sqlda );
      m_sqlda = (XSQLDA*) memAlloc( XSQLDA_LENGTH(count) );
      m_sqlda->version = SQLDA_VERSION1;
      m_sqlda->sqln = count;
      isc_dsql_describe_bind( status, &stmt, 1, m_sqlda );
   }
}


void FBSqlData::describeOut( isc_stmt_handle stmt )
{
   ISC_STATUS status[20];

   if( isc_dsql_describe( status, &stmt, 1, m_sqlda ) != 0 )
   {
      DBIHandleFB::throwError( __LINE__, FALCON_DBI_ERROR_BIND_INTERNAL, status, true );
   }

   if ( m_sqlda->sqld > m_sqlda->sqln )
   {
      int count = m_sqlda->sqld;
      memFree( m_sqlda );
      m_sqlda = (XSQLDA*) memAlloc( XSQLDA_LENGTH(count) );
      m_sqlda->version = SQLDA_VERSION1;
      m_sqlda->sqln = count;
      isc_dsql_describe( status, &stmt, 1, m_sqlda );
   }
}

void FBSqlData::allocOutput()
{
   m_bOwnBuffers = true;
   m_indicators = (ISC_SHORT*) memAlloc( sizeof(ISC_SHORT) * varCount() );

   for( int i = 0; i < varCount(); ++i )
   {
      XSQLVAR* v = var(i);
      v->sqldata = (ISC_SCHAR*) memAlloc( v->sqllen );
      v->sqlind = m_indicators + i;
      *v->sqlind = 0;
   }
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBIHandleFB::DBIHandleFB():
      m_bCommitted( false )
{
   m_pConn = 0;
   m_pTrans = 0;

}

DBIHandleFB::DBIHandleFB( const isc_db_handle &conn ):
      m_bCommitted( false )
{
   m_pConn = new FBConnRef( conn );
   m_pTrans = 0;
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
     throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
   }

   return m_pConn->handle();
}



void DBIHandleFB::options( const String& params )
{
   if( ! m_settings.parse( params ) )
   {
      // autocommit status is read on query
      throw new DBIError( ErrorParam( FALCON_DBI_ERROR_OPTPARAMS, __LINE__ )
            .extra( params ) );
   }
}

const DBISettingParams* DBIHandleFB::options() const
{
   return &m_settings;
}


DBIRecordset *DBIHandleFB::query( const String &sql, ItemArray* params )
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
      throwError( __LINE__, FALCON_DBI_ERROR_NOMEM, status, false );
   }

   AutoCString asQuery( sql );

   // call the query
   FBSqlData* out_tab = 0;

   ISC_STATUS res;
   try
   {
      out_tab = new FBSqlData;
      if( isc_dsql_prepare( status, &tr1, &stmt, asQuery.length(), asQuery.c_str(), 1, out_tab->table() ) )
      {
         throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status, true );
      }

      if( params == 0 )
      {
         res = isc_dsql_execute( status, &tr1, &stmt, 1, 0 ) != 0;
      }
      else
      {
         FBInBind bindings(stmt);
         bindings.bind( *params );
         res = isc_dsql_execute( status, &tr1, &stmt, 1, bindings.table() );
      }

      if( res != 0 )
      {
         throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status, true );
      }

      out_tab->describeOut( stmt );
      if( out_tab->varCount() != 0 )
      {
         return new DBIRecordsetFB( this, m_pTrans, stmt, out_tab );
      }
      else
      {
         delete out_tab;
         if( isc_dsql_free_statement( status, &stmt, DSQL_drop ) != 0 )
         {
            throwError( __LINE__, FALCON_DBI_ERROR_QUERY, status, false );
         }

         return 0;
      }
   }
   catch( ... )
   {
      delete out_tab;
      isc_dsql_free_statement( status, &stmt, DSQL_drop );
      throw;
   }

   //m_nLastAffected = (int64) nRowCount;
}



DBIStatement* DBIHandleFB::prepare( const String &query )
{
   isc_db_handle handle = getConnData();
   return 0;
   // TODO
}


int64 DBIHandleFB::getLastInsertedId( const String& sequenceName )
{
   isc_db_handle handle = getConnData();
   return 0;
   // TODO
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
      throwError( __LINE__, FALCON_DBI_ERROR_TRANSACTION, status, false );
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


void DBIHandleFB::throwError( int line, int code, ISC_STATUS* status, bool dsql )
{
   String desc;
   ISC_SCHAR msgBuffer[512];

   // For ib/fb, dynamic sql errors are different from engine errors.
   if( dsql )
   {
      // first write the error code,
      ISC_LONG SQLSTATUS = isc_sqlcode( status );
      desc.N((int64)SQLSTATUS).A(": ");

      // then the description.
      isc_sql_interprete( SQLSTATUS, msgBuffer, 512 );
      desc += msgBuffer;
   }
   else
   {
      // Get the main error
      ISC_STATUS* ep;
      ep = status;
      isc_interprete( msgBuffer, &ep );
      desc += msgBuffer;

      // Write the secondary errors as a list of [...; ...; ...] messages
      bool bDone = false;
      while( isc_interprete( msgBuffer, &ep ) )
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
   }

   throw new DBIError( ErrorParam( code, line ).extra(desc) );
}


} /* namespace Falcon */

/* end of fbsql_mod.cpp */


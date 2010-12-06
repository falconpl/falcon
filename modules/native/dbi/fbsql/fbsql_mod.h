/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_mod.h
 *
 * FB driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 16:58:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_FBSQL_H
#define FALCON_FBSQL_H

#include <falcon/dbi_common.h>
#include <falcon/srv/dbi_service.h>

#include <ibase.h>
#include <ib_util.h>
#include <iberror.h>


namespace Falcon
{

/** Class keeping SQLDA structures. */
class FBSqlData
{
public:
   FBSqlData();
   ~FBSqlData();

   void describeIn( isc_stmt_handle stmt );
   void describeOut( isc_stmt_handle stmt );

   /** Free allocated resources. */
   void release();
   void allocOutput();

   XSQLDA* table() const { return m_sqlda; }
   XSQLVAR* var(int n) { return m_sqlda->sqlvar + n; }
   int varCount() { return m_sqlda->sqld; }

private:
   XSQLDA* m_sqlda;
   ISC_SHORT* m_indicators;
   bool m_bOwnBuffers;
};


class FBConnRef: public DBIRefCounter<isc_db_handle> {
public:

   FBConnRef( const isc_db_handle& hDb ):
      DBIRefCounter<isc_db_handle>( hDb )
   {}

   virtual ~FBConnRef() {
      ISC_STATUS status[20];
      isc_detach_database( status, &handle() );
   }
};


class FBTransRef: public DBIRefCounter<isc_tr_handle> {
public:

   FBTransRef( const isc_tr_handle& hStmt ):
      DBIRefCounter<isc_tr_handle>( hStmt ),
      m_bClosed( false )
   {}

   virtual ~FBTransRef();
   void commit();
   void rollback();
   void commitRetaining();

   bool isClosed() const { return m_bClosed; }

private:
   bool m_bClosed;
};



class FBStmtRef: public DBIRefCounter<isc_stmt_handle> {
public:

   FBStmtRef( isc_stmt_handle hStmt ):
      DBIRefCounter<isc_stmt_handle>( hStmt )
   {}

   virtual ~FBStmtRef() {
      ISC_STATUS status[20];
      isc_dsql_free_statement( status, &handle(), DSQL_drop );
   }
};


class FBInBind: public DBIInBind
{

public:
   FBInBind( isc_stmt_handle stmt );
   virtual ~FBInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   XSQLDA* table() const { return m_data.table(); }
private:
   FBSqlData m_data;
   isc_stmt_handle m_stmt;
};




class DBIHandleFB;

class DBIRecordsetFB: public DBIRecordset
{
public:
   DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, isc_stmt_handle stmt, FBSqlData* data );
   DBIRecordsetFB( DBIHandleFB *dbt, FBTransRef* tref, FBStmtRef* sref, FBSqlData* data );
   virtual ~DBIRecordsetFB();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );

   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool fetchRow();
   virtual bool discard( int64 ncount );
   virtual void close();

protected:
   int m_nRow;
   int m_nRowCount;
   int m_nColumnCount;

   FBTransRef* m_tref;
   FBStmtRef* m_sref;
   FBSqlData* m_data;
};


class DBIStatementFB : public DBIStatement
{
protected:
   isc_stmt_handle m_statement;
   FBStmtRef* m_pStmt;
   FBConnRef* m_pConn;

   FBInBind* m_inBind;

public:
   DBIStatementFB( DBIHandleFB *dbh, const isc_stmt_handle& stmt );
   virtual ~DBIStatementFB();

   virtual DBIRecordset*  execute( ItemArray* params );
   virtual void reset();
   virtual void close();
};


class DBIHandleFB : public DBIHandle
{

public:
   DBIHandleFB();
   DBIHandleFB( const isc_db_handle &conn );
   virtual ~DBIHandleFB();

   virtual void options( const String& params );
   virtual const DBISettingParams* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, ItemArray* params = 0 );
   virtual DBIStatement* prepare( const String &query );
   virtual int64 getLastInsertedId( const String& name = "" );

   virtual void begin();
   virtual void commit();
   virtual void rollback();

   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result );

   // Checks for the db to be open and alive before proceed
   isc_db_handle getConnData();

   FBConnRef* connRef() const { return m_pConn; }
   FBTransRef* transRef() const { return m_pTrans; }

   // Throws a DBI error, using the last error code and description.
   static void throwError( int line, int code, ISC_STATUS* status, bool dsql );

private:
   FBConnRef* m_pConn;
   FBTransRef* m_pTrans;
   DBISettingParams m_settings;
   bool m_bCommitted;
};




class DBIServiceFB : public DBIService
{
public:
   DBIServiceFB() : DBIService( "DBI_fbsql" ) {}

   virtual void init();
   virtual DBIHandle *connect( const String &parameters );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

extern DBIServiceFB theFirebirdService;

}

#endif /* FALCON_FB_H */

/* end of fbsql_mod.h */


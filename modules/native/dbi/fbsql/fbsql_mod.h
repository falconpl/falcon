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

#ifndef FALCON_DBI_FBSQL_MOD_H
#define FALCON_DBI_FBSQL_MOD_H

#include <falcon/dbi_common.h>

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
   FBInBind( isc_db_handle dbh, isc_tr_handle tr, isc_stmt_handle stmt );
   virtual ~FBInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   XSQLDA* table() const { return m_data.table(); }
   ISC_QUAD createBlob( byte* data, int64 size );
private:
   FBSqlData m_data;
   isc_db_handle m_dbh;
   isc_tr_handle m_tr;
   isc_stmt_handle m_stmt;
   ISC_SHORT* m_sqlInd;
   ISC_QUAD* m_GIDS;
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

   FBConnRef* m_dbref;
   FBTransRef* m_tref;
   FBStmtRef* m_sref;
   FBSqlData* m_data;

   String* fetchBlob( ISC_QUAD *bId );
};


class DBIStatementFB : public DBIStatement
{
protected:
   isc_stmt_handle m_statement;
   FBTransRef* m_pTref;
   FBStmtRef* m_pStmt;
   FBConnRef* m_pConn;

   FBSqlData* m_outData;
   FBInBind* m_inBind;

   bool m_bAutoCommit;
   bool m_bGetAffected;

public:
   DBIStatementFB( DBIHandleFB *dbh, FBTransRef* pTref, const isc_stmt_handle& stmt, FBSqlData* outData );
   virtual ~DBIStatementFB();

   virtual DBIRecordset*  execute( ItemArray* params );
   virtual void reset();
   virtual void close();
};


class DBISettingParamsFB: public DBISettingParams
{
public:
   DBISettingParamsFB();
   DBISettingParamsFB( const DBISettingParamsFB& other );
   virtual ~DBISettingParamsFB();
   virtual bool parse( const String& connStr );

   /** Read affected rows after each query operation ( defaults to true ) */
   bool m_bGetAffected;

private:
   String m_sGetAffected;
};

class DBIHandleFB : public DBIHandle
{

public:
   DBIHandleFB(const Class* h);
   DBIHandleFB(const Class* h, const isc_db_handle &conn );
   virtual ~DBIHandleFB();

   virtual void connect( const String& params );
   virtual void options( const String& params );
   virtual const DBISettingParamsFB* options() const;
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
   static void throwError( int line, int code, ISC_STATUS* status );

   static int64 getAffected( isc_stmt_handle stmt, int etype = FALCON_DBI_ERROR_QUERY );

private:
   FBConnRef* m_pConn;
   FBTransRef* m_pTrans;
   DBISettingParamsFB m_settings;
   bool m_bCommitted;

   isc_stmt_handle internal_prepare( const String& query );
};

}

#endif /* FALCON_FB_H */

/* end of fbsql_mod.h */


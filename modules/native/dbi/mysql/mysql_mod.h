/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_mod.h
 *
 * MySQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 16:58:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_DBI_MYSQL_H_
#define _FALCON_DBI_MYSQL_H_

#include <falcon/dbi_common.h>
#include <falcon/dbi_service.h>

#include <mysql.h>

namespace Falcon
{


class MyDBIInBind: public DBIInBind
{

public:
   MyDBIInBind( MYSQL_STMT* stmt );

   virtual ~MyDBIInBind();

   virtual void onFirstBinding( int size );
   virtual void onItemChanged( int num );

   MYSQL_BIND* mybindings() const { return m_mybind; }

private:
   MYSQL_BIND* m_mybind;
   MYSQL_STMT* m_stmt;
};


class MyDBIOutBind: public DBIOutBind
{
public:
   MyDBIOutBind():
      bIsNull( false ),
      nLength( 0 )
   {}

   ~MyDBIOutBind() {}

   my_bool bIsNull;
   unsigned long nLength;
};

class DBIHandleMySQL;

class DBIRecordsetMySQL: public DBIRecordset
{
protected:
   int m_row;
   int m_rowCount;
   int m_columnCount;

   MYSQL_RES *m_res;
   MYSQL_FIELD* m_fields;

   bool m_bCanSeek;

public:
   DBIRecordsetMySQL( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL();

   virtual int64 getRowIndex();
   virtual int64 getRowCount();
   virtual int getColumnCount();
   virtual bool getColumnName( int nCol, String& name );
   virtual void close();
};

class DBIStatementMySQL;

class DBIRecordsetMySQL_STMT: public DBIRecordsetMySQL
{
protected:
   DBIStatementMySQL *m_stmt;

   // Binding data
   MYSQL_BIND* m_pMyBind;
   MyDBIOutBind* m_pOutBind;

   // used to keep track of blobs that must be zeroed before fetch
   int* m_pBlobId;
   int m_nBlobCount;

public:
   DBIRecordsetMySQL_STMT( DBIHandleMySQL *dbt, MYSQL_RES *res, DBIStatementMySQL *stmt, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_STMT();

   void init();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
   virtual void close();

   /** This cind of recordsets can generate a next recordset. */
   virtual DBIRecordset* getNext();

   virtual void gcMark( uint32 mark );
};


class DBIRecordsetMySQL_RES : public DBIRecordsetMySQL
{
protected:
   MYSQL_ROW m_rowData;
   void makeTimestamp( const String& str, Item& target );

public:
   DBIRecordsetMySQL_RES( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_RES();

   virtual bool fetchRow();
   virtual bool getColumnValue( int nCol, Item& value );
   virtual bool discard( int64 ncount );
};

class DBIRecordsetMySQL_RES_STR: public DBIRecordsetMySQL_RES
{
public:
   DBIRecordsetMySQL_RES_STR( DBIHandleMySQL *dbt, MYSQL_RES *res, bool bCanSeek = false );
   virtual ~DBIRecordsetMySQL_RES_STR();

   virtual bool getColumnValue( int nCol, Item& value );
};


class DBIHandleMySQL : public DBIHandle
{
protected:
   MYSQL *m_conn;
   DBISettingParams m_settings;

   MYSQL_STMT* my_prepare( const String &query, bool bCanFallback = false );
   int64 my_execute( MYSQL_STMT* stmt, MyDBIInBind& bindings, ItemArray* params );

public:
   DBIHandleMySQL( const Class* h );
   virtual ~DBIHandleMySQL();

   virtual void connect( const String& options );
   virtual void options( const String& params );
   virtual const DBISettingParams* options() const;
   virtual void close();

   virtual DBIRecordset *query( const String &sql, ItemArray* params );
   virtual DBIStatement* prepare( const String &query );
   virtual int64 getLastInsertedId( const String& name = "" );

   virtual void begin();
   virtual void commit();
   virtual void rollback();

   virtual void selectLimited( const String& query,
         int64 nBegin, int64 nCount, String& result );

   // Throws a DBI error, using the last error code and description.
   void throwError( const char* file, int line, int code );

   MYSQL* mysql() const { return m_conn; }
};


class DBIStatementMySQL : public DBIStatement
{
protected:
   MYSQL_STMT* m_statement;
   MyDBIInBind* m_inBind;
   bool m_bBound;
   bool m_owned;

public:
   DBIStatementMySQL( DBIHandleMySQL *dbh, MYSQL_STMT* stmt );
   virtual ~DBIStatementMySQL();

   virtual DBIRecordset* execute( ItemArray* params );
   virtual void reset();
   virtual void close();

   DBIHandleMySQL* getMySql() const { return static_cast<DBIHandleMySQL*>( m_dbh ); }
   MYSQL_STMT* my_statement() const { return m_statement; }
   void owned( bool w ) { m_owned = w; }
   bool owned() const { return m_owned; }
};


class DBIServiceMySQL : public DBIService
{
public:
   DBIServiceMySQL(Module* owner) : DBIService( FALCON_DBI_HANDLE_SERVICE_NAME, owner ) {}

   virtual DBIHandle *connect( const String &parameters );
};

}

#endif /* DBI_MYSQL_H */

/* end of mysql_mod.h */


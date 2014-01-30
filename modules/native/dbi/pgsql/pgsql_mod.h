/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_mod.h
 *
 * Pgsql driver main module interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar, Stanislas Marquis
 * Begin: Sun Dec 23 21:36:20 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef PGSQL_MOD_H
#define PGSQL_MOD_H

#include <libpq-fe.h>
#include <falcon/dbi_common.h>

namespace Falcon
{

class DBIHandlePgSQL;


int32 dbi_pgsqlQuestionMarksToDollars( const String& input, String& output );


class PgSQLHandlerRef: public DBIRefCounter<PGconn*>
{
public:
   PgSQLHandlerRef( PGconn* conn ):
      DBIRefCounter<PGconn*>( conn )
      {}

   ~PgSQLHandlerRef() {
      PQfinish( handle() );
   }
};


class DBIRecordsetPgSQL
    :
    public DBIRecordset
{
protected:

    int64       m_row;
    int64       m_rowCount;
    int         m_columnCount;

    PGresult*   m_res;
    PgSQLHandlerRef* m_pConn;

public:

    DBIRecordsetPgSQL( DBIHandlePgSQL* dbh, PGresult* res );
    virtual ~DBIRecordsetPgSQL();

    virtual bool fetchRow();
    virtual int64 getRowIndex();
    virtual int64 getRowCount();
    virtual int getColumnCount();
    virtual bool getColumnName( int nCol, String& name );
    virtual bool getColumnValue( int nCol, Item& value );
    virtual bool discard( int64 ncount );
    virtual void close();

};


class DBIStatementPgSQL
    :
    public DBIStatement
{
protected:

    uint32  m_nParams;
    String  m_execString;
    String  m_name;
    PgSQLHandlerRef* m_pConn;

    void getExecString( uint32 nParams, const String& name );

public:

    DBIStatementPgSQL( DBIHandlePgSQL* dbh );
    virtual ~DBIStatementPgSQL();

    virtual DBIRecordset* execute( ItemArray* params = 0 );
    virtual void reset();
    virtual void close();

    void init( const String& query, const String& name="happy_falcon" );
};


class DBIHandlePgSQL
    :
    public DBIHandle
{
protected:

    PGconn* m_conn;
    bool    m_bInTrans;
    DBISettingParams m_settings;

    PgSQLHandlerRef* m_pConn;

public:

    DBIHandlePgSQL( const Class* h, PGconn* = 0 );
    virtual ~DBIHandlePgSQL();

    PGconn* getConn() { return m_conn; }

    virtual void connect( const String &parameters );
    virtual void options( const String& params );
    virtual const DBISettingParams* options() const;

    virtual void begin();
    virtual void commit();
    virtual void rollback();

    virtual DBIRecordset* query( const String &sql, ItemArray* params = 0 );
    virtual DBIStatement* prepare( const String &query );
    virtual DBIStatement* prepareNamed( const String &name, const String& query );
    virtual int64 getLastInsertedId( const String& name = "" );

    virtual void selectLimited( const String& query, int64 nBegin, int64 nCount, String& result );

    virtual void close();

    static void throwError( const char* file, int line, PGresult* res );
    PGresult* internal_exec( const String& sql, int64& affectedRows );

    PgSQLHandlerRef* getConnRef() const { return m_pConn; }
};


} // !namespace Falcon

#endif /* PGSQL_MOD_H */

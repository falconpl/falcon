/*
   FALCON - The Falcon Programming Language.
   FILE: pgsql.h
   
   Pgsql driver main module interface
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: Sun Dec 23 21:36:20 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef DBI_PGSQL_H
#define DBI_PGSQL_H

#include <libpq-fe.h>

#include "../include/dbiservice.h"

namespace Falcon
{

class DBIRecordsetPgSQL : public DBIRecordset
{
public:
   virtual dbr_status next();
   virtual dbr_status fetch( CoreArray *resultCache );
   virtual dbr_status fetchColumns( CoreArray *resultCache );
   virtual int fetchRowCount();
   virtual int fetchColumnCount();
   virtual void close();
   virtual dbr_status getLastError( String &description );
};

class DBITransactionPgSQL : public DBITransaction
{
public:
   virtual DBIRecordset *query( const String &query, dbt_status &retval );
   virtual int execute( const String &query, dbt_status &retval );
   virtual dbt_status commit();
   virtual dbt_status rollback();
   virtual void close();
   virtual dbt_status getLastError( String &description );
};

class DBIHandlePgSQL : public DBIHandle
{
private:
   PGconn *m_conn;
   
public:
   DBIHandlePgSQL() { m_conn = NULL; }
   DBIHandlePgSQL( PGconn *conn ) { m_conn = conn; }
   virtual ~DBIHandlePgSQL() {}
   
   DBITransaction *startTransaction();
   dbh_status closeTransaction( DBITransaction *tr );
   virtual dbh_status getLastError( String &description );
   virtual dbh_status close();
};

class DBIServicePgSQL : public DBIService
{
public:
   DBIServicePgSQL() : DBIService( "DBI_pgsql" ) {}
   
   virtual dbi_status init();
   virtual DBIHandle *connect( const String &parameters, bool persistent, 
                               dbi_status &retval, String &errorMessage );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );
};

}

extern Falcon::DBIServicePgSQL thePgSQLService;

#endif /* DBI_PGSQL_H */

/* end of pgsql.h */


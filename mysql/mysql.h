/*
   FALCON - The Falcon Programming Language.
   FILE: mysql.h

   Mysql driver main module interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 20:33:57 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef DBI_MYSQL_H
#define DBI_MYSQL_H

#include "../include/dbiservice.h"

namespace Falcon
{
class DBITransactionMysql: public DBITransaction
{
public:
   virtual dbt_status query(const String &query );
   virtual dbt_status fetch( CoreArray *resultCache );
   virtual dbt_status fetchColums( CoreArray *resultCache );
   virtual dbt_status commit();
   virtual dbt_status rollback();
   virtual void close();
   virtual dbt_status getLastError( String &description );

   // you may add mysql specific functions here, but they have to be virtual.
};


class DBIHandleMysql: public DBIHandle
{
public:

   DBIHandleMysql() {}
   virtual ~DBIHandleMysql() {}
   DBITransaction *startTransaction();
   dbh_status closeTransaction( DBITransaction *tr );
   virtual dbh_status getLastError( String &description );
   virtual dbh_status close();

   // you may add mysql specific functions here, but they have to be virtual.
};


class DBIServiceMySQL: public DBIService
{
public:
   DBIServiceMySQL():
      DBIService( "DBI_mysql" )
   {}

   virtual dbi_status init();
   virtual DBIHandle *connect( const String &parameters, bool persistent, dbi_status &retval );
   virtual dbi_status getLastError( String &description );
   virtual CoreObject *makeInstance( VMachine *vm, DBIHandle *dbh );

   // you may add mysql specific functions here, but they have to be virtual.
};

}

// finally, the DLL wide object that manages interaction with MySQL:
extern Falcon::DBIServiceMySQL theMySQLService;

#endif

/* end of mysql.h */


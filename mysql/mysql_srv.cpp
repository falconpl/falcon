/*
   FALCON - The Falcon Programming Language.
   FILE: mysql_srv.cpp

   MySQL Falcon service/driver
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 23:16:37 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/engine.h>
#include "mysql.h"

namespace Falcon {

/**********************************************************
   Transaction class
**********************************************************/

DBITransactionMysql::dbt_status DBITransactionMysql::query(const String &query )
{
   return s_ok;
}


DBITransactionMysql::dbt_status DBITransactionMysql::fetch( CoreArray *resultCache )
{
   return s_ok;
}


DBITransactionMysql::dbt_status DBITransactionMysql::fetchColums( CoreArray *resultCache )
{
   return s_ok;
}


DBITransactionMysql::dbt_status DBITransactionMysql::commit()
{
   return s_ok;
}


DBITransactionMysql::dbt_status DBITransactionMysql::rollback()
{
   return s_ok;
}


void DBITransactionMysql::close()
{
}


DBITransactionMysql::dbt_status DBITransactionMysql::getLastError( String &description )
{
   return s_ok;
}

/**********************************************************
   DB Handler class
**********************************************************/

DBITransaction *DBIHandleMysql::startTransaction()
{
   new DBITransactionMysql;
}

DBIHandleMysql::dbh_status DBIHandleMysql::closeTransaction( DBITransaction *tr )
{
   return s_ok;
}

DBIHandleMysql::dbh_status DBIHandleMysql::getLastError( String &description )
{
   return s_ok;
}

DBIHandleMysql::dbh_status DBIHandleMysql::close()
{
   return s_ok;
}



/**********************************************************
   Main service class class
**********************************************************/



DBIServiceMySQL::dbi_status DBIServiceMySQL::init()
{
   // nothing to do
   return s_ok;
}


DBIHandle *DBIServiceMySQL::connect( const String &parameters, bool persistent, dbi_status &retval )
{
   // for now, nothing to do
   return new DBIHandleMysql;
}

DBIServiceMySQL::dbi_status DBIServiceMySQL::getLastError( String &description )
{
   return s_ok;
}

// this is pretty interesting
CoreObject *DBIServiceMySQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   // we have to create an instance of MySQL class and configure it properly.
   Item *clMysql = vm->findGlobalItem( "MySQL" );
   if ( clMysql == 0 || ! clMysql->isClass() || clMysql->asClass()->symbol()->name() != "MySQL" )
   {
      //Someone passed us a VM without or with a wrong MySql class.
      // It would be a good way to make us crash, as we'd put a MySQL user data in a non
      // mySql object, that may use the data in funny ways.

      // raise an error.
      return 0;
   }

   CoreObject *myobj = clMysql->asClass()->createInstance();
   myobj->setUserData( dbh );
   return myobj;
}

}

/* end of mysql_srv.cpp */


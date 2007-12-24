/*
   FALCON - The Falcon Programming Language.
   FILE: pgsql_srv.cpp
   
   PgSQL Falcon service/driver
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: Sun Dec 23 21:54:42 2007
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/engine.h>
#include "pgsql.h"

namespace Falcon
{

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransaction::dbt_status DBITransactionPgSQL::query( const String &query )
{
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::fetch( CoreArray *resultCache )
{
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::fetchColumns( CoreArray *resultCache )
{
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::commit()
{
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::rollback()
{
   return s_ok;
}

void DBITransactionPgSQL::close()
{
}

DBITransaction::dbt_status DBITransactionPgSQL::getLastError( String &description )
{
   return s_ok;
}

/******************************************************************************
 * DB Handler class
 *****************************************************************************/

DBITransaction *DBIHandlePgSQL::startTransaction()
{
   return new DBITransactionPgSQL;
}

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::closeTransaction( DBITransaction *tr )
{
   return s_ok;
}

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::getLastError( String &description )
{
   return s_ok;
}

DBIHandlePgSQL::dbh_status DBIHandlePgSQL::close()
{
   return s_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

DBIServicePgSQL::dbi_status DBIServicePgSQL::init()
{
   return s_ok;
}

DBIHandle *DBIServicePgSQL::connect( const String &parameters, bool persistent, dbi_status &retval )
{
   return new DBIHandlePgSQL;
}

DBIServicePgSQL::dbi_status DBIServicePgSQL::getLastError( String &description )
{
   return s_ok;
}

CoreObject *DBIServicePgSQL::makeInstance( VMachine *vm, DBIHandle *dbh )
{
   Item *cl = vm->findGlobalItem( "PgSQL" );
   if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "PgSQL" )
   {
      // TODO: raise an error.
      return 0;
   }

   CoreObject *obj = cl->asClass()->createInstance();
   obj->setUserData( dbh );
   
   return obj;
}

} /* namespace Falcon */

/* end of pgsql_srv.cpp */


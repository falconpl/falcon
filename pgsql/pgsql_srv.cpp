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

#include <string.h>

#include <falcon/engine.h>
#include "pgsql.h"

namespace Falcon
{

/******************************************************************************
 * Recordset class
 *****************************************************************************/

DBIRecordset::dbr_status DBIRecordsetPgSQL::next()
{
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::fetch( CoreArray *resultCache )
{
   return s_ok;
}

DBIRecordset::dbr_status DBIRecordsetPgSQL::fetchColumns( CoreArray *resultCache )
{
   return s_ok;
}

int DBIRecordsetPgSQL::fetchRowCount()
{
   return 0;
}

int DBIRecordsetPgSQL::fetchColumnCount()
{
   return 0;
}

void DBIRecordsetPgSQL::close()
{
}

DBIRecordsetPgSQL::dbr_status DBIRecordsetPgSQL::getLastError( String &description )
{
   return s_ok;
}

/******************************************************************************
 * Transaction class
 *****************************************************************************/

DBITransactionPgSQL::DBITransactionPgSQL( DBIHandle *dbh ) 
    : DBITransaction( dbh )
{
   m_inTransaction = false;
}

DBIRecordset *DBITransactionPgSQL::query( const String &query, dbt_status &retval )
{
   retval = s_not_implemented;
   return NULL;
}

DBITransaction::dbt_status DBITransactionPgSQL::begin()
{
   m_inTransaction = true;
   
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::commit()
{
   m_inTransaction = false;
   
   return s_ok;
}

DBITransaction::dbt_status DBITransactionPgSQL::rollback()
{
   m_inTransaction = false;
   return s_ok;
}

void DBITransactionPgSQL::close()
{
   if ( m_inTransaction )
   {
      commit();
   }
   
   m_inTransaction = false;
   
   m_dbh->closeTransaction( this );
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
   DBITransactionPgSQL *t = new DBITransactionPgSQL( this );
   if ( t->begin() != DBITransaction::s_ok )
   {
      // TODO: set error state
      
      delete t;
      
      return NULL;
   }
   
   return t;
}

/******************************************************************************
 * Transaction Handler Class
 *****************************************************************************/

DBIHandlePgSQL::DBIHandlePgSQL()
{
   m_conn = NULL;
   m_connTr = NULL;
}

DBIHandlePgSQL::DBIHandlePgSQL( PGconn *conn )
{
   m_conn = conn;
   m_connTr = NULL;
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
   if ( m_conn != NULL )
   {
      PQfinish( m_conn );
      m_conn = NULL;
   }
   
   return s_ok;
}

/******************************************************************************
 * Main service class
 *****************************************************************************/

DBIServicePgSQL::dbi_status DBIServicePgSQL::init()
{
   return s_ok;
}

DBIHandle *DBIServicePgSQL::connect( const String &parameters, bool persistent, 
                                     dbi_status &retval, String &errorMessage )
{
   AutoCString connParams( parameters );
   PGconn *conn = PQconnectdb( connParams.c_str () );
   if ( conn == NULL ) {
      retval = s_memory_alloc_error;
      return NULL;
   }
   
   if ( PQstatus( conn ) != CONNECTION_OK ) {
      retval = s_connect_failed;
      // TODO: Use append? I used append because copy and = were causing memory
      // errors because the memory PQerrorMessage is pointing to is free'd when
      // the later PQfinish is called. I would have thought that .copy() would
      // have taken care of this, but I suffered the same corrupt string
      // symptoms with copy as I did =. = with a strdup worked fine, but I was
      // then worried about memory leaks.
      errorMessage.append( PQerrorMessage( conn ) );
      errorMessage.remove( errorMessage.length() - 1, 1 ); // Get rid of newline
      
      PQfinish( conn );
      
      return NULL;
   }
   
   retval = s_ok;
   return new DBIHandlePgSQL( conn );
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


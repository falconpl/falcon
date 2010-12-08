/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_mod.cpp
 *
 * Oracle Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>
#include <errmsg.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "oracle_mod.h"

namespace Falcon
{
/******************************************************************************
 * DB Handler Class
 *****************************************************************************/
DBIHandleOracle::~DBIHandleOracle()
{
    DBIHandleOracle::close();
}  

void DBIHandleOracle::close()
{
    if ( o_conn != NULL )
    {
        o_pConn->decref();
        // Kill the connection then the environment
        o_env->terminateConnection( o_conn );
        terminateEnv( o_env );
    }
}

void DBIHandleOracle::terminateEnv()
{
    if ( o_env != NULL)
    {
        terminateEnivronment( o_env );
    }
}
  
void DBIHandleOracle::commit()
{
    if( o_conn == NULL )
    {
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
    }

    if( oracle_query( o_conn, "COMMIT" ) != 0 )
    {
        throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_TRANSACTION );
    }
}

void DBIHandleOracle::rollback()
{
    if( o_conn == NULL )
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
    
    if( oracle_query( o_conn, "ROLLBACK" ) != 0 )
    {
        throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_TRANSACTION );
    }
}


ORACLE_STMT* DBIHandleOracle::o_prepare( const String &query )
{
    if( o_conn == NULL )
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
    
    Statement* stmt;
    Statement::Status* status = stmt->status();
    
    if( stmt == 0 || status == 0 )
    {
        throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_NOMEM );
    }
    
    AutoCString cquery( query );
    try
    {
        stmt->setSQL( cquery.c_str() );
    }
    catch
    {
        throwError( __FILE__, __LINE__, FALCON_DBI_ERROR_QUERY );
    }

    // TODO There might be more we need to add here.    
    
    return stmt;
}

DBIStatement* DBIHandleOracle::prepare( const String &query )
{
    Statement stmt;
    Statement::Status status = stmt->status();

    return new DBIStatementOracle( this, stmt, status );
}

/******************************************************************************
 * Main service class
 *****************************************************************************/
void DBIServiceOracle::init()
{
}

DBIHandle *DBIServiceOracle::connect( const String &parameters )
{
    ORACLE *conn = oracle_init( NULL );
    ORACLE *env = oracle_init( NULL );

    if ( conn = NULL )
    {
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
    }

    DBIConnParams connParams;

    if( ! connParams.parse( parameters ) )
    {
        oracle_close( conn );
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
                .extra( parameters ) );
    }

    env = Environment::createEnvironment();
    
    // We'll need at least a username, password, and db.
    conn = env->createConnection(connParams.o_szUser, pconnParams.o_szPassword, connParams.o_szConn);
    
    if ( conn == NULL )
    {
        String errorMessage = getMessage();
        errorMessage.bufferize();
        oracle_close( conn );
        
        throw new DBIError( ErrorParam( en, __LINE__).extra( errorMessage ) );
    }
    
    
#if ( OCCI_MAJOR_VERSION > 9 )
    env->setCacheSortedFlush( true );
#endif
    return new DBIHandleOracle( conn );    
}

CoreObject *DBIServiceOracle::makeInstance( VMachine *vm, DBIHandle *dbh )
{
    Item *cl = vm->findWKI( "Oracle" );
    if ( cl == 0 || ! cl->isClass() )
    {
        throw new DBIError( ErrorParam( FALCON_DBI_ERROR_INVALID_DRIVER, __LINE__ ) );
    }
    
    CoreObject *obj = cl->asClass()->createInstance();
    obj->setUserData( dbh );
    
    return obj;
}

}/* namespace Falcon */

/* end of oracle_mod.cpp */


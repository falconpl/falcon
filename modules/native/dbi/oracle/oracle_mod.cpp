/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_mod.cpp
 *
 * Oracle Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <string.h>
#include <stdio.h>

#include <falcon/engine.h>
#include <falcon/dbi_error.h>
#include "oracle_mod.h"

namespace Falcon
{
    /******************************************************************************
     * Transaction Class
     *****************************************************************************/
    DBIStatementOracle::DBIStatementOracle( DBIHandleOracle *dbh, oracle::occi::Statement* stmt ):
        DBIStatement( dbh ),
        o_statement( stmt ),
        o_inBind(0), 
        o_bBound( false )
    {
        //o_pConn = dbh->getConn();
        //o_pConn->incref();
        //o_pStmt = new ORACLEStmtHandle( stmt );
    }


    DBIStatementOracle::~DBIStatementOracle()
    {
        close();
    }

    DBIRecordset* DBIStatementOracle::execute( ItemArray* params )
    {
        if( o_statement == 0 )
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_STMT, __LINE__ ) );

        /*   
         * TODO  
         *
         */

        return 0;

    }

    void DBIStatementOracle::close()
    {
        if ( o_statement != 0 )
        {
            o_statement = 0;
            delete o_inBind;
            o_inBind = 0;
            //o_pConn->decref();
            //o_pStmt->decref();
        }
    }

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
            //o_pConn->decref();
            // Kill the connection then the environment
            o_env->terminateConnection( o_conn );
            //oracle::occi::Environment::terminateEnvironment( o_env );
        }
    }

    void DBIHandleOracle::commit()
    {
        if( o_conn == NULL )
        {
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
        }
    }

    void DBIHandleOracle::rollback()
    {
        if( o_conn == NULL )
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CLOSED_DB, __LINE__ ) );
    }

    /******************************************************************************
     * Main service class
     *****************************************************************************/
    void DBIServiceOracle::init()
    {
    }

    DBIHandle *DBIServiceOracle::connect( const String &parameters )
    {

        oracle::occi::Connection *conn;
        oracle::occi::Environment *env;

        if ( conn = NULL )
        {
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_NOMEM, __LINE__) );
        }

        DBIConnParams connParams;

        if( ! connParams.parse( parameters ) )
        {
            //close( conn );
            throw new DBIError( ErrorParam( FALCON_DBI_ERROR_CONNPARAMS, __LINE__)
                    .extra( parameters ) );
        }

        /*
           env = oracle::occi::Environment::createEnvironment();

        // We'll need at least a username, password, and db.    
        conn = env->createConnection(connParams.o_szUser, connParams.o_szPassword, connParams.o_szConn);
        */
        if ( conn == NULL )
        {
            String errorMessage = oracle::occi::SQLException::getMessage();
            errorMessage.bufferize();
            //close( conn ); 
            
            int en = mysql_errno( conn ) == ER_BAD_DB_ERROR ?
               FALCON_DBI_ERROR_DB_NOTFOUND : FALCON_DBI_ERROR_CONNECT;

            throw new DBIError( ErrorParam( en, __LINE__).extra( errorMessage ) );
        }
        

#if ( OCCI_MAJOR_VERSION > 9 )
        env->setCacheSortedFlush( true );
#endif
        //return new DBIHandleOracle( conn );    
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


/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_srv.cpp
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
#include "oracle_mod.h"

namespace Falcon
{
    /******************************************************************************
     * DB Handler class
     *****************************************************************************/
    DBIHandleOracle::~DBIHandleOracle()
    {
        DBIHandleOracle::close();
    }

    DBITransaction *DBIHandleOracle::startTransaction()
    {
        DBITransactionOracle *t = new DBITransactionOracle( this );
        if ( t->begin() != dbi_ok ) {
            // TODO: filter useful information to the script level
            delete t;

            return NULL;
        }

        return t;
    }

    DBITransaction* DBIHandleOracle::getDefaultTransaction()
    {
        return m_connTr == NULL ? (m_connTr = new DBITransactionOracle( this )): m_connTr;
    }

    DBIHandleMySQL::DBIHandleOracle()
    {
        o_conn = NULL;
        o_connTr = NULL;
    }

    DBIHandleMySQL::DBIHandleOracle( Connection *conn )
    {
        o_conn = conn;
        o_connTr = NULL;
    }

    dbi_status DBIHandleOracle::closeTransaction( DBITransaction *tr )
    {
        return dbi_ok;
    }

    int64 DBIHandleOracle::getLastInsertedId()
    {
        return oracle_insert_id( o_conn );
    }

    int64 DBIHandleOracle::getLastInsertedId( const String& sequenceName )
    {
        return oracle_insert_id( o_conn );
    }

    dbi_status DBIHandleMySQL::getLastError( String &description )
    {
        if ( o_conn == NULL )
            return dbi_invalid_connection;

        const char *errorMessage = getMessage();
        if ( errorMessage == NULL )
            return dbi_no_error_message;

        description.bufferize( errorMessage );

        return dbi_ok;
    }

    dbi_status DBIHandleOracle::escapeString( const String &value, String &escaped )
    {
        if ( value.length() == 0 )
            return dbi_ok;

        AutoCString asValue( value );

        int maxLen = ( value.length() * 2 ) + 1;
        char *cTo = (char *) memAlloc( sizeof( char ) * maxLen );

        // TODO
        size_t convertedSize = mysql_real_escape_string( o_conn, cTo,
                asValue.c_str(), asValue.length() );

        if ( convertedSize < value.length() ) {
            memFree( cTo );
            return dbi_error;
        }

        escaped.fromUTF8( cTo );
        memFree( cTo );

        return dbi_ok;
    }

    dbi_status DBIHandleOracle::close()
    {
        if ( m_conn != NULL ) {

            m_conn = NULL;
        }

        return dbi_ok;
    }

    /******************************************************************************
     * Main service class
     *****************************************************************************/

    dbi_status DBIServiceMySQL::init()
    {
        return dbi_ok;
    }

    DBIHandle *DBIServiceOracle::connect( const String &parameters, bool persistent,
            dbi_status &retval, String &errorMessage )
    {
        char *host, *user, *passwd, *db, *port, *unixSocket, *clientFlags;
        unsigned int iPort, iClientFlag;
        char** vals[] = { &host, &user, &passwd, &db, &port, &unixSocket, &clientFlags, 0 };

        Environment *env;
        Connection *conn;

        AutoCString asConnParams( parameters );
        char* connp = (char*) asConnParams.c_str();
        char*** elem = vals;
        char* beg = connp;

        // tokenize ","
        while( *connp && *elem )
        {
            if ( *connp == ',' )
            {
                if ( beg == connp )
                {
                    // empty "," ?
                    **elem = 0;
                }
                else
                    **elem = beg;

                ++elem;
                *connp = '\0';
                ++connp;
                beg = connp;
            }
            else
                ++connp;
        }

        // get the last element
        if ( *elem )
        {
            **elem = beg;
            ++elem;
            // nulls remaining parameters.
            while( *elem )
            {
                **elem = 0;
                ++elem;
            }
        }

        env = Environment::createEnvironment(); 
        conn = env->createConnection(user, passwd);

        if ( conn == NULL )
        {
            errorMessage = getMessage();
            errorMessage.bufferize();
            terminateConnection( conn );
            terminateEnvironment( env );
            retval = dbi_connect_error;
            return NULL;
        }
        else
        {
            retval = dbi_ok;
            return new DBIHandleOracle( conn );
        }

    }

    CoreObject *DBIServiceOracle::makeInstance( VMachine *vm, DBIHandle *dbh )
    {
        Item *cl = vm->findWKI( "Oracle" );
        if ( cl == 0 || ! cl->isClass() || cl->asClass()->symbol()->name() != "Oracle" )
        {
            throw new DBIError( ErrorParam( dbi_driver_not_found, __LINE__ )
                    .desc( "Oracle DBI driver was not found" ) );
            return 0;
        }

        CoreObject *obj = cl->asClass()->createInstance();
        obj->setUserData( dbh );

        return obj;
    }
}/* namespace Falcon */

/* end of oracle_srv.cpp */


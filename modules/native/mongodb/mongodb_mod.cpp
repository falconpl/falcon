/*
 *  Falcon MongoDB - Internal
 */

#include "mongodb_mod.h"

#include <stdio.h>//debug..
#include <string.h>

#include <falcon/engine.h>

namespace Falcon
{
namespace MongoDB
{

void
ConnRef::decref()
{
    if ( --mCnt <= 0 )
    {
        if ( mConn )
        {
            mongo_destroy( mConn );
            free( mConn );
        }
        delete this;
    }
}

Connection::Connection( const char* host, int port, mongo_connection* conn )
    :
    mConn( 0 )
{
    hostPort( host, port );
    if ( conn )
        mConn = new ConnRef( conn );
}

Connection::~Connection()
{
    if ( mConn )
    {
        mConn->decref();
        mConn = 0;
    }
}

void
Connection::gcMark( uint32 )
{
}

FalconData*
Connection::clone() const
{
    return 0;
}

void
Connection::hostPort( const char* host, int port )
{
    if ( host )
    {
        memset( mOptions.host, 0, 255 );
        strncpy( mOptions.host, host, 254 );
    }
    if ( port > 0 )
        mOptions.port = port;
}

void
Connection::options( mongo_connection_options* options )
{
    if ( !options )
        return;

    memcpy( &mOptions, options, sizeof( mongo_connection_options ) );
}

int
Connection::connect()
{
    mongo_connection* conn = 0;
    mongo_conn_return ret;

    if ( mConn ) // existing conn object
    {
        conn = mConn->conn();
        if ( conn->connected ) // reset connection
        {
            mongo_disconnect( conn );
        }
        ret = mongo_reconnect( conn );
    }
    else // new conn
    {
        conn = (mongo_connection*) malloc( sizeof( mongo_connection ) );
        if ( !conn ) // no mem
            return -1;
        memset( conn, 0, sizeof( mongo_connection ) );
        // connect first time
        ret = mongo_connect( conn, &mOptions );
        if ( ret == 0 ) // success
            mConn = new ConnRef( conn );
        else
            free( conn );
    }

    return (int) ret;
}

int
Connection::disconnect()
{
    if ( !mConn )
        return 0;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return 0;

    bson_bool_t b = mongo_disconnect( conn );
    return b ? 1 : 0;
}

bool
Connection::isConnected() const
{
    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    return conn->connected ? true : false;
}

bool
Connection::authenticate( const char* db,
                          const char* user,
                          const char* pass )
{
    if ( !db || db[0] == '\0'
        || !user || user[0] == '\0'
        || !pass || pass[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    bson_bool_t ret = mongo_cmd_authenticate( conn, db, user, pass );
    return ret ? true : false;
}

bool
Connection::addUser( const char* db,
                     const char* user,
                     const char* pass )
{
    if ( !db || db[0] == '\0'
        || !user || user[0] == '\0'
        || !pass || pass[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    mongo_cmd_add_user( conn, db, user, pass );
    return true;
}

bool
Connection::dropDatabase( const char* db )
{
    if ( !db || db[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    bson_bool_t ret = mongo_cmd_drop_db( conn, db );
    return ret ? true : false;
}


} // !namespace MongoDB
} // !namespace Falcon

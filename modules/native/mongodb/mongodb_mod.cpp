/*
 *  Falcon MongoDB - Internal
 */

#include "mongodb_mod.h"

#include <stdio.h>//debug..
#include <string.h>

#include <falcon/engine.h>
#include <falcon/autocstring.h>

namespace Falcon
{
namespace MongoDB
{

/*******************************************************************************
    ConnRef class
*******************************************************************************/

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

/*******************************************************************************
    Connection class
*******************************************************************************/

BSONObj emptyBSONObj;

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

bool
Connection::dropCollection( const char* db,
                            const char* coll )
{
    if ( !db || db[0] == '\0'
        || !coll || coll[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    bson_bool_t ret = mongo_cmd_drop_collection( conn, db, coll, NULL );
    return ret ? true : false;
}

/*******************************************************************************
    BSONObj class
*******************************************************************************/

static bson mEmpty;

BSONObj::BSONObj( const int bytesNeeded )
{
    // prepare internal buffer
    bson_buffer_init( &mBuf );
    if ( bytesNeeded > 0 )
    {
        bson_ensure_space( &mBuf, bytesNeeded );
    }
    // object always born as 'empty'
    bson_empty( &mObj );
    mFinalized = true;
}

BSONObj::~BSONObj()
{
    // clear buffer
    bson_buffer_destroy( &mBuf );
    // clear bson object
    bson_destroy( &mObj );
}

void
BSONObj::gcMark( uint32 )
{
}

FalconData*
BSONObj::clone() const
{
    return 0;
}

bson*
BSONObj::empty()
{
    static bool done = false;
    if ( !done )
    {
        bson_empty( &mEmpty );
        done = true;
    }
    return &mEmpty;
}

BSONObj*
BSONObj::append( const char* nm )
{
    bson_append_null( &mBuf, nm );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm, const int i )
{
    bson_append_int( &mBuf, nm, i );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm, const int64_t il )
{
    bson_append_long( &mBuf, nm, il );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm, const double d )
{
    bson_append_double( &mBuf, nm, d );
    if ( mFinalized ) mFinalized = false;
    return this;
}


BSONObj*
BSONObj::append( const char* nm, const char* str )
{
    bson_append_string( &mBuf, nm, str );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm, const Falcon::String& str )
{
    AutoCString zStr( str );
    bson_append_string( &mBuf, nm, zStr.c_str() );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm, const bool b )
{
    bson_append_bool( &mBuf, nm, b );
    if ( mFinalized ) mFinalized = false;
    return this;
}

bson*
BSONObj::finalize()
{
    if ( mFinalized )
        return &mObj;

    bson_destroy( &mObj );
    bson_from_buffer( &mObj, &mBuf );
    mFinalized = true;
    return &mObj;
}

void
BSONObj::reset( const int bytesNeeded )
{
    // reset buffer
    bson_buffer_destroy( &mBuf );
    bson_buffer_init( &mBuf );
    if ( bytesNeeded > 0 )
    {
        bson_ensure_space( &mBuf, bytesNeeded );
    }
    // reset bson object
    bson_destroy( &mObj );
    bson_empty( &mObj );
}

} // !namespace MongoDB
} // !namespace Falcon

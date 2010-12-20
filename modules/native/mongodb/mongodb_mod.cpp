/*
 *  Falcon MongoDB - Internal
 */

#include "mongodb_mod.h"

#include <stdio.h>//debug..
#include <string.h>

#include <falcon/autocstring.h>
#include <falcon/engine.h>
#include <falcon/iterator.h>


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

Connection::Connection( const char* host,
                        int port,
                        mongo_connection* conn )
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
Connection::hostPort( const char* host,
                      int port )
{
    if ( host )
    {
        if ( !strcmp( host, "localhost" ) )
            host = "127.0.0.1";

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
    if ( !conn->connected )
        return false;

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
    if ( !conn->connected )
        return false;

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
    if ( !conn->connected )
        return false;

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
    if ( !conn->connected )
        return false;

    bson_bool_t ret = mongo_cmd_drop_collection( conn, db, coll, NULL );
    return ret ? true : false;
}

bool
Connection::insert( const char* ns,
                    BSONObj* data )
{
    if ( !ns || ns[0] == '\0'
        || !data )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    mongo_insert( conn, ns, data->finalize() );
    return true;
}

bool
Connection::insert( const String& ns,
                    BSONObj* data )
{
    if ( ns.length() == 0 || !data )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    AutoCString zNs( ns );

    mongo_insert( conn, zNs.c_str(), data->finalize() );
    return true;
}

bool
Connection::insert( const char* ns,
                    const CoreArray& data )
{
    if ( !ns || ns[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    const uint32 sz = data.length();
    if ( sz == 0 ) // nothing to insert
        return true;

    // check items are bson objects
    const Item* it;
    for ( uint32 i=0; i < sz; ++i )
    {
        it = &data.at( i );
        if ( !it->isObject() || !it->asObjectSafe()->derivedFrom( "BSON" ) )
            return false;
    }
    // insert all
    BSONObj* bobj;
    for ( uint32 i=0; i < sz; ++i )
    {
        it = &data.at( i );
        bobj = static_cast<BSONObj*>( it->asObjectSafe()->getUserData() );
        if ( !insert( ns, bobj ) )
            return false;
    }
    return true;
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

BSONObj::BSONObj( const bson* bobj )
{
    // prepare a buffer
    bson_buffer_init( &mBuf );
    // copy given bson
    bson_copy( &mObj, bobj );
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
BSONObj::genOID( const char* nm )
{
    bson_append_new_oid( &mBuf, nm );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_null( buf, nm );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const int i,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_int( buf, nm, i );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const int64_t il,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_long( buf, nm, il );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const double d,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_double( buf, nm, d );
    if ( mFinalized ) mFinalized = false;
    return this;
}


BSONObj*
BSONObj::append( const char* nm,
                 const char* str,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_string( buf, nm, str );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const Falcon::String& str,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    AutoCString zStr( str );
    bson_append_string( buf, nm, zStr.c_str() );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const bool b,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_bool( buf, nm, b );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const CoreArray& array,
                 bson_buffer* parentBuf )
{
    // we need to check items validity before changing the buffer!
    // that's why this is not public

    if ( parentBuf == 0 )
        parentBuf = &mBuf;

    const uint32 sz = array.length();
    bson_buffer* sub = bson_append_start_array( parentBuf, nm );

    if ( sz == 0 ) // empty array
    {
        bson_append_finish_object( sub );
        if ( mFinalized ) mFinalized = false;
        return this;
    }

    for ( uint32 i=0; i < sz; ++i )
    {
        Item it = array.at( i );
        append( "0", it, sub );
    }

    bson_append_finish_object( sub );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const CoreDict& dict,
                 bson_buffer* parentBuf )
{
    // we need to check items validity before changing the buffer!
    // that's why this is not public

    if ( parentBuf == 0 )
        parentBuf = &mBuf;

    bson_buffer* sub = bson_append_start_object( parentBuf, nm );

    if ( dict.length() == 0 ) // empty dict
    {
        bson_append_finish_object( sub );
        if ( mFinalized ) mFinalized = false;
        return this;
    }

    Iterator iter( (Sequence*) &dict.items() );

    while ( iter.hasCurrent() )
    {
        Item key = iter.getCurrentKey();
        Item val = iter.getCurrent();
        AutoCString k( key );
        append( k.c_str(), val, sub );
        iter.next();
    }

    bson_append_finish_object( sub );
    if ( mFinalized ) mFinalized = false;
    return this;
}

bool
BSONObj::append( const char* nm,
                 const Falcon::Item& item,
                 bson_buffer* buf,
                 const bool doCheck )
{
    switch ( item.type() )
    {
    case FLC_ITEM_NIL:
        return append( nm, buf );
    case FLC_ITEM_INT:
        return append( nm, (int64_t) item.asInteger(), buf );
    case FLC_ITEM_BOOL:
        return append( nm, item.asBoolean(), buf );
    case FLC_ITEM_NUM:
        return append( nm, item.asNumeric(), buf );
    case FLC_ITEM_STRING:
        return append( nm, *item.asString(), buf );
    case FLC_ITEM_ARRAY:
        if ( doCheck && !arrayIsSupported( *item.asArray() ) )
            return false;
        return append( nm, *item.asArray(), buf );
    case FLC_ITEM_DICT:
        if ( doCheck && !dictIsSupported( *item.asDict() ) )
            return false;
        return append( nm, *item.asDict(), buf );
    default: // unsupported type
        return false;
    }
}

int
BSONObj::appendMany( const CoreDict& dict )
{
    if ( dict.length() == 0 ) // nothing to append
        return 0;

    // we want to check data before updating bson buffer
    Iterator iter( (Sequence*) &dict.items() );
    Item* k, *v;

    while ( iter.hasCurrent() )
    {
        k = &iter.getCurrentKey();
        if ( !k->isString() ) // bad key
            return 1;
        v = &iter.getCurrent();
        if ( !itemIsSupported( *v ) ) // bad value
            return 2;
        iter.next();
    }
    // really appending data
    iter.goTop();
    while ( iter.hasCurrent() )
    {
        k = &iter.getCurrentKey();
        v = &iter.getCurrent();
        AutoCString key( *k );
        append( key.c_str(), *v, 0, false ); // not checking
        iter.next();
    }
    return 0;
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
    if ( !mFinalized ) mFinalized = true;
}

bool
BSONObj::itemIsSupported( const Falcon::Item& item )
{
    switch ( item.type() )
    {
    case FLC_ITEM_NIL:
    case FLC_ITEM_INT:
    case FLC_ITEM_BOOL:
    case FLC_ITEM_NUM:
    case FLC_ITEM_STRING:
        return true;
    case FLC_ITEM_ARRAY:
        return arrayIsSupported( *item.asArray() );
    case FLC_ITEM_DICT:
        return dictIsSupported( *item.asDict() );
    default:
        return false;
    }
}

bool
BSONObj::arrayIsSupported( const CoreArray& array )
{
    const uint32 sz = array.length();
    if ( sz == 0 )
        return true;

    for ( uint32 i=0; i < sz; ++i )
    {
        Item it = array.at( i );
        if ( !itemIsSupported( it ) )
            return false;
    }
    return true;
}

bool
BSONObj::dictIsSupported( const CoreDict& dict )
{
    if ( dict.length() == 0 )
        return true;

    Iterator iter( (Sequence*) &dict.items() );

    while ( iter.hasCurrent() )
    {
        Item k = iter.getCurrentKey();
        if ( !k.isString() )
            return false;

        Item v = iter.getCurrent();
        if ( !itemIsSupported( v ) )
            return false;

        iter.next();
    }
    return true;
}

/*******************************************************************************
    BSONIter class
*******************************************************************************/

BSONIter::BSONIter( BSONObj* data )
    :
    mCurrentType( -1 )
{
    bson_copy( &mData, data->finalize() );
    bson_iterator_init( &mIter, mData.data );
}

BSONIter::BSONIter( const bson* data )
    :
    mCurrentType( -1 )
{
    bson_copy( &mData, data );
    bson_iterator_init( &mIter, mData.data );
}

BSONIter::~BSONIter()
{
    bson_destroy( &mData );
}

void
BSONIter::gcMark( uint32 )
{
}

FalconData*
BSONIter::clone() const
{
    return 0;
}

void
BSONIter::reset()
{
    bson_iterator_init( &mIter, mData.data );
    mCurrentType = -1;
}

bool
BSONIter::next()
{
    mCurrentType = (int) bson_iterator_next( &mIter );
    return mCurrentType == bson_eoo ? false : true;
}

const char*
BSONIter::currentKey()
{
    return mCurrentType > 0 ? bson_iterator_key( &mIter ): 0;
}

Falcon::Item*
BSONIter::currentValue()
{
    if ( mCurrentType <= 0 )
        return 0;

    return makeItem( (bson_type) mCurrentType, &mIter );
}

Falcon::Item*
BSONIter::makeItem( const bson_type tp,
                    bson_iterator* iter )
{
    Item* it = 0;

    switch ( tp )
    {
    case bson_double:
        it = new Item( bson_iterator_double_raw( iter ) );
        break;
    case bson_string:
        it = new Item( bson_iterator_string( iter ) );
        break;
    case bson_object:
    {
        bson_iterator iter2;
        bson_iterator_subiterator( iter, &iter2 );
        it = makeObject( &iter2 );
        break;
    }
    case bson_array:
    {
        bson_iterator iter2;
        bson_iterator_subiterator( iter, &iter2 );
        it = makeArray( &iter2 );
        break;
    }
    case bson_bindata:
        //...
        break;
    case bson_undefined:
        it = new Item( bson_iterator_value( iter ) );
        break;
    case bson_oid:
    {
        char id[25];
        bson_oid_to_string( bson_iterator_oid( iter ), id );
        it = new Item( id );
        break;
    }
    case bson_bool:
        it = new Item();
        it->setBoolean( (bool) bson_iterator_bool_raw( iter ) );
        break;
    case bson_date:
        //...
        break;
    case bson_null:
        it = new Item();
        break;
    case bson_regex:
        //...
        break;
    case bson_dbref: // deprecated
        //...
        break;
    case bson_symbol:
        it = new Item( bson_iterator_string( iter ) );
        break;
    case bson_codewscope:
        it = new Item( bson_iterator_code( iter ) );
        break;
    case bson_int:
        it = new Item( bson_iterator_int_raw( iter ) );
        break;
    case bson_timestamp:
        //...
        break;
    case bson_long:
        it = new Item( bson_iterator_long_raw( iter ) );
        break;
    case bson_eoo:
    default:
        return it;
    }

    return it;
}

Falcon::Item*
BSONIter::makeArray( bson_iterator* iter )
{
    CoreArray* arr = new CoreArray;

    while ( bson_iterator_next( iter ) != bson_eoo )
    {
        Item* v = makeItem( bson_iterator_type( iter ), iter );
        arr->append( *v );
    }

    return new Item( arr );
}

Falcon::Item*
BSONIter::makeObject( bson_iterator* iter )
{
    CoreDict* dict = new CoreDict( new LinearDict );

    while( bson_iterator_next( iter ) != bson_eoo )
    {
        Item* k = new Item( bson_iterator_key( iter ) );
        Item* v = makeItem( bson_iterator_type( iter ), iter );
        dict->put( *k, *v );
    }

    return new Item( dict );
}


} // !namespace MongoDB
} // !namespace Falcon

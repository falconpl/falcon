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

bool
Connection::update( const char* ns,
                    BSONObj* cond,
                    BSONObj* op,
                    const bool upsert,
                    const bool multiple )
{
    if ( !ns || ns[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    int flg = 0;
    if ( upsert )
        flg |= MONGO_UPDATE_UPSERT;
    if ( multiple )
        flg |= MONGO_UPDATE_MULTI;

    mongo_update( conn, ns, cond->finalize(), op->finalize(), flg );
    return true;
}

bool
Connection::remove( const char* ns,
                    BSONObj* cond )
{
    if ( !ns || ns[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    mongo_remove( conn, ns, cond->finalize() );
    return true;
}

bool
Connection::findOne( const char* ns,
                     BSONObj* query,
                     BSONObj** ret )
{
     if ( !ns || ns[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    bson res;
    bson_bool_t b = mongo_find_one( conn, ns,
                                    query ? query->finalize() : BSONObj::empty(),
                                    NULL,
                                    ret ? &res : NULL );

    if ( b && ret )
    {
        *ret = new BSONObj( &res );
        bson_destroy( &res );
    }

    return b ? true : false;
}

bool
Connection::find( const char* ns,
                  BSONObj* query,
                  BSONObj* fields,
                  const int skip,
                  const int limit,
                  CoreArray** res )
{
    if ( !ns || ns[0] == '\0' )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    bson* quer = query ? query->finalize() : BSONObj::empty();

    mongo_cursor* cur = mongo_find( conn, ns,
                                    quer,
                                    fields ? fields->finalize() : NULL,
                                    limit, skip, 0 );

    if ( res )
    {
        *res = new CoreArray;
        Item* wki = VMachine::getCurrent()->findWKI( "BSON" );
        while ( mongo_cursor_next( cur ) )
        {
            CoreObject* obj = wki->asClass()->createInstance();
            BSONObj* bobj = new BSONObj( &cur->current );
            obj->setUserData( bobj );
            (*res)->append( obj );
        }
    }

    mongo_cursor_destroy( cur );
    return true;
}

int64
Connection::count( const char* db,
                   const char* coll,
                   BSONObj* query )
{
    if ( !db || db[0] == '\0'
        || !coll || coll[0] == '\0' )
        return -1;

    if ( !mConn )
        return -1;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return -1;

    return mongo_count( conn, db, coll,
                        query ? query->finalize() : BSONObj::empty() );
}

bool
Connection::command( const char* db,
                     BSONObj* cmd,
                     BSONObj** res )
{
    if ( !db || db[0] == '\0' || !cmd )
        return false;

    if ( !mConn )
        return false;

    mongo_connection* conn = mConn->conn();
    if ( !conn->connected )
        return false;

    bson out;
    bson_bool_t b = mongo_run_command( conn, db, cmd->finalize(), &out );

    if ( b && res )
        *res = new BSONObj( &out );

    return b ? true : false;
}

/*******************************************************************************
    ObjectID class
*******************************************************************************/

ObjectID::ObjectID( const CoreClass* cls )
    :
    CoreObject( cls )
{
    bson_oid_gen( &mOID );
}

ObjectID::ObjectID( const CoreClass* cls,
                    const char* str )
    :
    CoreObject( cls )
{
    fromString( str );
}

ObjectID::ObjectID( const CoreClass* cls,
                    const bson_oid_t* oid )
    :
    CoreObject( cls ),
    mOID( *oid )
{
}

ObjectID::ObjectID( const ObjectID& other )
    :
    CoreObject( other ),
    mOID( other.mOID )
{
}

ObjectID::~ObjectID()
{
}

bool
ObjectID::getProperty( const String& nm,
                       Item& it ) const
{
    return defaultProperty( nm, it );
}

bool
ObjectID::setProperty( const String& nm,
                       const Item& it )
{
    return false;
}

Falcon::CoreObject*
ObjectID::clone() const
{
    return new ObjectID( *this );
}

Falcon::CoreObject*
ObjectID::factory( const CoreClass* cls, void*, bool )
{
    return new ObjectID( cls );
}

bool
ObjectID::fromString( const char* str )
{
    if ( strlen( str ) != 24 )
        return false;
    bson_oid_from_string( &mOID, str );
    return true;
}

const char*
ObjectID::toString()
{
    bson_oid_to_string( &mOID, mStr );
    return mStr;
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
    if ( !mBuf.finished )
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
                 const bson_oid_t* oid )
{
    bson_append_oid( &mBuf, nm, oid );
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
BSONObj::appendDate( const char* nm,
                     const bson_date_t date,
                     bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_date( buf, nm, date );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const Falcon::TimeStamp& ts,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    TimeStamp epoch( 1970, 1, 1, 0, 0, 0, 0, tz_UTC );
    epoch.distance( ts );

    const bson_date_t t =
        (int64) epoch.m_msec +
        ( (int64) epoch.m_second * 1000 ) +
        ( (int64) epoch.m_minute * 60 * 1000 ) +
        ( (int64) epoch.m_hour * 60 * 60 * 1000 ) +
        ( (int64) epoch.m_day * 24 * 60 * 60 * 1000 );

    bson_append_date( buf, nm, t );
    if ( mFinalized ) mFinalized = false;
    return this;
}

BSONObj*
BSONObj::append( const char* nm,
                 const Falcon::MemBuf& mem,
                 bson_buffer* buf )
{
    if ( !buf )
        buf = &mBuf;

    bson_append_binary( buf, nm, mem.wordSize(), (const char*)mem.data(), mem.length() );
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
    case FLC_ITEM_MEMBUF:
        return append( nm, *item.asMemBuf(), buf );
    case FLC_ITEM_ARRAY:
        if ( doCheck && !arrayIsSupported( *item.asArray() ) )
            return false;
        return append( nm, *item.asArray(), buf );
    case FLC_ITEM_DICT:
        if ( doCheck && !dictIsSupported( *item.asDict() ) )
            return false;
        return append( nm, *item.asDict(), buf );
    case FLC_ITEM_OBJECT:
    {
        CoreObject* obj = item.asObjectSafe();
        if ( obj->derivedFrom( "ObjectID" ) )
        {
            ObjectID* oid = static_cast<ObjectID*>( obj );
            return append( nm, oid->oid() );
        }
        else
        if ( obj->derivedFrom( "TimeStamp" ) )
        {
            TimeStamp* ts = static_cast<TimeStamp*>( obj->getUserData() );
            return append( nm, *ts );
        }
        return false;
    }
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

int
BSONObj::createFromDict( const CoreDict& dict,
                         BSONObj** bobj )
{
    fassert( bobj );
    *bobj = new BSONObj;
    if ( !*bobj )
        return -1;
    return (*bobj)->appendMany( dict );
}

bson*
BSONObj::finalize()
{
    if ( mFinalized )
        return &mObj;

    bson_destroy( &mObj );
    bson_from_buffer( &mObj, &mBuf ); // that 'finishes' the buffer
    mFinalized = true;
    return &mObj;
}

void
BSONObj::reset( const int bytesNeeded )
{
    // reset buffer
    if ( !mBuf.finished )
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
BSONObj::hasKey( const char* key )
{
    if ( !key || key[0] == '\0' )
        return false;

    bson_iterator iter;
    bson_iterator_init( &iter, finalize()->data );

    while ( bson_iterator_next( &iter ) != bson_eoo )
    {
        if ( !strcmp( key, bson_iterator_key( &iter ) ) )
            return true;
    }
    return false;
}

Falcon::Item*
BSONObj::value( const char* key )
{
    if ( !key || key[0] == '\0' )
        return 0;

    bson_iterator iter;
    bson_iterator_init( &iter, finalize()->data );
    bson_type tp;

    while ( ( tp = bson_iterator_next( &iter ) ) != bson_eoo )
    {
        if ( !strcmp( key, bson_iterator_key( &iter ) ) )
        {
            return BSONIter::makeItem( tp, &iter );
        }
    }
    return 0;
}

Falcon::CoreDict*
BSONObj::asDict()
{
    bson_iterator iter;
    bson_iterator_init( &iter, finalize()->data );
    CoreDict* dict = new CoreDict( new LinearDict );
    bson_type tp;
    const char* k;
    Item* v;

    while ( ( tp = bson_iterator_next( &iter ) ) != bson_eoo )
    {
        k = bson_iterator_key( &iter );
        v = BSONIter::makeItem( tp, &iter );
        dict->put( String( k ), *v );
    }

    return dict;
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
    case FLC_ITEM_MEMBUF:
        return true;
    case FLC_ITEM_ARRAY:
        return arrayIsSupported( *item.asArray() );
    case FLC_ITEM_DICT:
        return dictIsSupported( *item.asDict() );
    case FLC_ITEM_OBJECT:
    {
        const CoreObject* obj = item.asObjectSafe();
        if ( obj->derivedFrom( "ObjectID" ) )
            return true;
        else
        if ( obj->derivedFrom( "TimeStamp" ) )
            return true;
        return false;
    }
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

bool
BSONIter::find( const char* nm )
{
    mCurrentType = bson_find( &mIter, &mData, nm );
    return mCurrentType != bson_eoo ? true : false;
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
    {
        const byte* ptr = (byte*) bson_iterator_bin_data( iter );
        const uint32 sz = bson_iterator_bin_len( iter );
        byte* data;
        MemBuf* mb = 0;
        switch ( bson_iterator_bin_type( iter ) )
        {
        case 4:
            data = (byte*) memAlloc( sz * 4 );
            memcpy( data, ptr, sz * 4 );
            mb = new MemBuf_4( data, sz, memFree );
            break;
        case 3:
            data = (byte*) memAlloc( sz * 3 );
            memcpy( data, ptr, sz * 3 );
            mb = new MemBuf_3( data, sz, memFree );
            break;
        case 2:
            data = (byte*) memAlloc( sz * 2 );
            memcpy( data, ptr, sz * 2 );
            mb = new MemBuf_2( data, sz, memFree );
            break;
        case 1:
            data = (byte*) memAlloc( sz * 1 );
            memcpy( data, ptr, sz * 1 );
            mb = new MemBuf_1( data, sz, memFree );
            break;
        default:
            break;
        }
        fassert( mb );
        it = new Item( mb );
        break;
    }
    case bson_undefined:
        it = new Item( bson_iterator_value( iter ) );
        break;
    case bson_oid:
    {
        VMachine* vm = VMachine::getCurrent();
        ObjectID* oid = new ObjectID( vm->findWKI( "ObjectID" )->asClass(),
                                      bson_iterator_oid( iter ) );
        it = new Item( oid );
        break;
    }
    case bson_bool:
        it = new Item();
        it->setBoolean( (bool) bson_iterator_bool_raw( iter ) );
        break;
    case bson_date:
    {
        int64 d, h, m, s, ms;
        const bson_date_t t = bson_iterator_date( iter );
        bson_date_t tt = llabs( t );

        d = t / ( 1000*60*60*24 );
        tt -= llabs( d ) * ( 1000*60*60*24 );
        h = tt / ( 1000*60*60 );
        tt -= h * ( 1000*60*60 );
        m = tt / ( 1000*60 );
        tt -= m * ( 1000*60 );
        s = tt / 1000;
        tt -= s * 1000;
        ms = tt;

        VMachine* vm = VMachine::getCurrent();
        Item* wki = vm->findWKI( "TimeStamp" );
        CoreObject* obj = wki->asClass()->createInstance();
        TimeStamp tmp( 0, 0, d, h, m, s, ms, tz_UTC );
        TimeStamp* ts = new TimeStamp( 1970, 1, 1, 0, 0, 0, 0, tz_UTC );
        ts->add( tmp );
        obj->setUserData( ts );
        it = new Item( obj );
        break;
    }
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

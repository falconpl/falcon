/*
 *  Falcon MongoDB - Internal stuff
 */

#ifndef MONGODB_MOD_H
#define MONGODB_MOD_H

#include <falcon/carray.h>
#include <falcon/falcondata.h>
#include <falcon/string.h>

#include "src/mongo.h"

namespace Falcon
{
namespace MongoDB
{

class BSONObj;


class ConnRef
{
public:

    ConnRef( mongo_connection* conn )
        :
        mCnt( 1 ),
        mConn( conn )
    {}

    void incref() { ++mCnt; }
    void decref();

    mongo_connection* conn() { return mConn; }

private:

    ~ConnRef() {}

    int mCnt;
    mongo_connection* mConn; // is malloc'ed
};


class Connection
    :
    public Falcon::FalconData
{
public:

    Connection( const char* host="127.0.0.1",
                int port=27017,
                mongo_connection* conn=0 );
    virtual ~Connection();

    virtual void gcMark( uint32 );
    virtual FalconData* clone() const;

    mongo_connection* conn() const { return mConn ? mConn->conn() : 0; }
    ConnRef* connRef() const { return mConn; }

    void hostPort( const char* host=0,
                   int port=0 );
    const char* host() const { return mOptions.host; }
    int port() const { return mOptions.port; }

    mongo_connection_options* options() { return &mOptions; }
    void options( mongo_connection_options* options );

    int connect();
    int disconnect();
    bool isConnected() const;
    bool authenticate( const char* db,
                       const char* user,
                       const char* pass );
    bool addUser( const char* db,
                  const char* user,
                  const char* pass );
    bool dropDatabase( const char* db );
    bool dropCollection( const char* db,
                         const char* coll );
    bool insert( const char* ns,
                 BSONObj* data );
    bool insert( const String& ns,
                 BSONObj* data );
    bool insert( const char* ns,
                 const CoreArray& data );

protected:

    mongo_connection_options    mOptions;
    ConnRef*                    mConn;

};


class BSONObj
    :
    public FalconData
{
public:

    BSONObj( const int bytesNeeded=0 );
    BSONObj( const bson* bobj ); // bobj is copied not owned
    virtual ~BSONObj();

    virtual void gcMark( uint32 );
    virtual FalconData* clone() const;

    bson_buffer* buffer() { return &mBuf; }
    bson* finalize();

    void reset( const int bytesNeeded=0 );

    BSONObj* genOID( const char* nm="_id" );
    BSONObj* append( const char* nm,
                     bson_buffer* buf=0 ); // append a null value
    BSONObj* append( const char* nm,
                     const int i,
                     bson_buffer* buf=0 );
    BSONObj* append( const char* nm,
                     const int64_t il,
                     bson_buffer* buf=0 );
    BSONObj* append( const char* nm,
                     const double d,
                     bson_buffer* buf=0 );
    BSONObj* append( const char* nm,
                     const char* str,
                     bson_buffer* buf=0 );
    BSONObj* append( const char* nm,
                     const Falcon::String& str,
                     bson_buffer* buf=0 );
    BSONObj* append( const char* nm,
                     const bool b,
                     bson_buffer* buf=0 );

    // Return true if item was successfuly appended.
    bool append( const char* nm,
                 const Falcon::Item& item,
                 bson_buffer* buf=0,
                 const bool doCheck=true ); // unless you know what you're doing

    int appendMany( const CoreDict& dict );

    // Return true if this item can be appended safely.
    static bool itemIsSupported( const Falcon::Item& item );
    // Return true if this array content is supported by our driver.
    static bool arrayIsSupported( const CoreArray& array );
    // Return true if this dict content is supported by our driver.
    static bool dictIsSupported( const CoreDict& dict );

    static bson* empty(); // helper

protected:

    BSONObj* append( const char* nm,
                     const CoreArray& array,
                     bson_buffer* parentBuf=0 );

    BSONObj* append( const char* nm,
                     const CoreDict& dict,
                     bson_buffer* parentBuf=0 );

    bson_buffer mBuf;
    bson        mObj;
    bool        mFinalized;

};


class BSONIter
    :
    public FalconData
{
public:

    BSONIter( BSONObj* data );
    BSONIter( const bson* data );
    virtual ~BSONIter();

    virtual void gcMark( uint32 );
    virtual FalconData* clone() const;

    void reset();

    bool next();
    const char* currentKey();
    Falcon::Item* currentValue();

protected:

    static Falcon::Item* makeItem( const bson_type tp,
                                   bson_iterator* iter );
    static Falcon::Item* makeArray( bson_iterator* iter );
    static Falcon::Item* makeObject( bson_iterator* iter );

    bson            mData;
    bson_iterator   mIter;
    int             mCurrentType;

};


} // !namespace MongoDB
} // !namespace Falcon

#endif // !MONGODB_MOD_H

/*
 *  Falcon MongoDB - Internal stuff
 */

#ifndef MONGODB_MOD_H
#define MONGODB_MOD_H

#include <falcon/falcondata.h>
#include <falcon/string.h>

#include "src/mongo.h"

namespace Falcon
{
namespace MongoDB
{


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

    mongo_connection* conn() const { return mConn ? mConn->conn() : 0; }
    ConnRef* connRef() const { return mConn; }

    virtual void gcMark( uint32 );
    virtual FalconData* clone() const;

    void hostPort( const char* host=0, int port=0 );
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
    virtual ~BSONObj();

    virtual void gcMark( uint32 );
    virtual FalconData* clone() const;

    bson_buffer* buffer() { return &mBuf; }
    bson* finalize();

    void reset( const int bytesNeeded=0 );

    BSONObj* append( const char* nm ); // append a null value
    BSONObj* append( const char* nm, const int i );
    BSONObj* append( const char* nm, const int64_t il );
    BSONObj* append( const char* nm, const double d );
    BSONObj* append( const char* nm, const char* str );
    BSONObj* append( const char* nm, const Falcon::String& str );
    BSONObj* append( const char* nm, const bool b );


    static bson* empty(); // helper

protected:

    bson_buffer mBuf;
    bson        mObj;
    bool        mFinalized;
};

} // !namespace MongoDB
} // !namespace Falcon

#endif // !MONGODB_MOD_H

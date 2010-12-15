/*
 *  Falcon MongoDB - Internal stuff
 */

#ifndef MONGODB_MOD_H
#define MONGODB_MOD_H

#include <falcon/falcondata.h>

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

    int connect();
    int disconnect();

protected:

    mongo_connection_options    mOptions;
    ConnRef*                    mConn;

};


} // !namespace MongoDB
} // !namespace Falcon

#endif // !MONGODB_MOD_H

/*
 *  Falcon DataMatrix - Service
 */

#define FALCON_EXPORT_SERVICE

#include "dmtx_srv.h"

#include "dmtx_mod.h"


namespace Falcon
{
#if 0
MongoDBService::MongoDBService()
    :
    Falcon::Service( MONGODB_SERVICENAME )
{
}

MongoDBService::~MongoDBService()
{
}

bool
MongoDBService::createConnection( const char* host,
                                  int port,
                                  mongo_connection* mongo_conn,
                                  FalconData** conn )
{
    if ( !conn )
        return false;
    *conn = 0;
    *conn = new Falcon::MongoDB::Connection( host, port, mongo_conn );
    if ( !*conn )
        return false;
    return true;
}

bool
MongoDBService::createBSONObj( const int bytesNeeded,
                               FalconData** bson )
{
    if ( !bson )
        return false;
    *bson = 0;
    *bson = new Falcon::MongoDB::BSONObj( bytesNeeded );
    if ( !*bson )
        return false;
    return true;
}
#endif
} // !namespace Falcon

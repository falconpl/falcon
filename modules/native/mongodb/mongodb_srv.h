/*
 *  Falcon MongoDB - Service
 */

#ifndef MONGODB_SRV_H
#define MONGODB_SRV_H

#include <falcon/service.h>

#include "src/mongo.h"

#define MONGODB_SERVICENAME     "MongoDB"

namespace Falcon
{

class FalconData;


class FALCON_SERVICE MongoDBService
    :
    public Falcon::Service
{
public:

    MongoDBService();
    virtual ~MongoDBService();

    /**
     *  \brief Create a MongoDB object.
     *  \param host The host
     *  \param port The port
     *  \param mongo_conn An existing (malloc'ed) mongo connection or NULL
     *  \param conn The returned FalconData object
     *  \return false on error
     */
    virtual bool createConnection( const char* host,
                                   int port,
                                   mongo_connection* mongo_conn,
                                   FalconData** conn );

    virtual bool createBSONObj( const int bytesNeeded,
                                FalconData** bson );
};

} // !namespace Falcon

#endif // !MONGODB_SRV_H

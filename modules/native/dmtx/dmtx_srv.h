/*
 *  Falcon DataMatrix - Service
 */

#ifndef DMTX_SRV_H
#define DMTX_SRV_H

#include <falcon/service.h>

#include <dmtx.h>

#define DMTX_SERVICENAME     "dmtx"

namespace Falcon
{
#if 0
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
#endif
} // !namespace Falcon

#endif // !DMTX_SRV_H

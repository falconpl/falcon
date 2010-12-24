/*
 *  Falcon MongoDB - Extension
 */

#ifndef MONGODB_EXT_H
#define MONGODB_EXT_H

#include <falcon/setup.h>

#include <falcon/error_base.h>
#include <falcon/error.h>

#ifndef FALCON_MONGODB_ERROR_BASE
#define FALCON_MONGODB_ERROR_BASE   16000
#endif

#define MONGODB_ERR_NOMEM           (FALCON_MONGODB_ERROR_BASE + 0)
#define MONGODB_ERR_CREATE_CONN     (FALCON_MONGODB_ERROR_BASE + 1)
#define MONGODB_ERR_CONNECT         (FALCON_MONGODB_ERROR_BASE + 2)
#define MONGODB_ERR_CREATE_BSON     (FALCON_MONGODB_ERROR_BASE + 3)

namespace Falcon
{

class VMachine;

namespace Ext
{

class MongoDBError
    :
    public ::Falcon::Error
{
public:

    MongoDBError()
        :
        Error( "MongoDBError" )
    {}

    MongoDBError( const ErrorParam &params )
        :
        Error( "MongoDBError", params )
    {}

};

FALCON_FUNC MongoDBError_init( VMachine* vm );

FALCON_FUNC MongoDBConnection_init( VMachine* vm );
FALCON_FUNC MongoDBConnection_host( VMachine* vm );
FALCON_FUNC MongoDBConnection_port( VMachine* vm );
FALCON_FUNC MongoDBConnection_connect( VMachine* vm );
FALCON_FUNC MongoDBConnection_disconnect( VMachine* vm );
FALCON_FUNC MongoDBConnection_isConnected( VMachine* vm );
FALCON_FUNC MongoDBConnection_authenticate( VMachine* vm );
FALCON_FUNC MongoDBConnection_addUser( VMachine* vm );
FALCON_FUNC MongoDBConnection_dropDatabase( VMachine* vm );
FALCON_FUNC MongoDBConnection_dropCollection( VMachine* vm );
FALCON_FUNC MongoDBConnection_insert( VMachine* vm );
FALCON_FUNC MongoDBConnection_update( VMachine* vm );
FALCON_FUNC MongoDBConnection_remove( VMachine* vm );
FALCON_FUNC MongoDBConnection_findOne( VMachine* vm );
FALCON_FUNC MongoDBConnection_find( VMachine* vm );
FALCON_FUNC MongoDBConnection_count( VMachine* vm );

FALCON_FUNC MongoOID_init( VMachine* vm );
FALCON_FUNC MongoOID_toString( VMachine* vm );

FALCON_FUNC MongoBSON_init( VMachine* vm );
FALCON_FUNC MongoBSON_reset( VMachine* vm );
FALCON_FUNC MongoBSON_genOID( VMachine* vm );
FALCON_FUNC MongoBSON_append( VMachine* vm );
FALCON_FUNC MongoBSON_asDict( VMachine* vm );
FALCON_FUNC MongoBSON_hasKey( VMachine* vm );
FALCON_FUNC MongoBSON_value( VMachine* vm );

FALCON_FUNC MongoBSONIter_init( VMachine* vm );
FALCON_FUNC MongoBSONIter_next( VMachine* vm );
FALCON_FUNC MongoBSONIter_key( VMachine* vm );
FALCON_FUNC MongoBSONIter_value( VMachine* vm );
FALCON_FUNC MongoBSONIter_reset( VMachine* vm );
FALCON_FUNC MongoBSONIter_find( VMachine* vm );

} // !namespace Ext
} // !namespace Falcon

#endif // !MONGODB_EXT_H

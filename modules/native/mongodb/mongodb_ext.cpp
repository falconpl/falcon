/*
 *  Falcon MongoDB - Extension
 */

#include "mongodb_ext.h"
#include "mongodb_mod.h"
#include "mongodb_srv.h"
#include "mongodb_st.h"

#include <falcon/engine.h>
#include <falcon/vm.h>

#include <stdio.h>//debug..

extern Falcon::MongoDBService theMongoDBService;

/*#
    @beginmodule mongodb
 */

namespace Falcon
{
namespace Ext
{

/*#
    @class MongoDBError
    @brief Error generated MongoDB operations.
    @optparam code The error code
    @optparam desc The description for the error code
    @optparam extra Extra information specifying the error conditions.
    @from Error( code, desc, extra )
*/
FALCON_FUNC MongoDBError_init( VMachine* vm )
{
    CoreObject *einst = vm->self().asObject();

    if ( einst->getUserData() == 0 )
        einst->setUserData( new MongoDBError );

    ::Falcon::core::Error_init( vm );
}


/*#
    @class MongoDB
    @brief Create a client connection to a MongoDB database.
    @optparam host default to localhost.
    @optparam port default to 27017.
 */
FALCON_FUNC MongoDBConnection_init( VMachine* vm )
{
    Item* i_host = vm->param( 0 );
    Item* i_port = vm->param( 1 );

    if ( ( i_host && !i_host->isString() )
        || ( i_port && !i_port->isInteger() ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "[S,I]" ) );
    }

    const char* host;
    AutoCString zHost;
    if ( i_host )
    {
        zHost.set( *i_host );
        host = zHost.c_str();
    }
    else
        host = "127.0.0.1";

    int port = i_port ? i_port->asInteger() : 27017;

    Falcon::MongoDB::Connection* conn;
    CoreObject* self = vm->self().asObjectSafe();

    if ( !theMongoDBService.createConnection( host, port, 0, (FalconData**)&conn ) )
    {
        throw new MongoDBError( ErrorParam( MONGODB_ERR_CREATE_CONN, __LINE__ )
                                .desc( FAL_STR( _err_create_conn ) ) );
    }
    self->setUserData( conn );
    vm->retval( self );
}

/*#
    @method host MongoDB
    @optparam host
    @brief When given a parameter, change the host for the next connection attempt. Else return the current host.
    @return self (with param) or the current host.
 */
FALCON_FUNC MongoDBConnection_host( VMachine* vm )
{
    Item* i_host = vm->param( 0 );

    if ( i_host && !i_host->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                              .extra( "[S]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    if ( i_host ) // set host
    {
        AutoCString zHost( *i_host );
        conn->hostPort( zHost.c_str() );
        vm->retval( self );
    }
    else // get host
    {
        String s( conn->host() );
        s.bufferize();
        vm->retval( s );
    }
}

/*#
    @method port MongoDB
    @optparam port
    @brief When given a parameter, change the port for the next connection attempt. Else return the current port.
    @return self (with param) or the current port.
 */
FALCON_FUNC MongoDBConnection_port( VMachine* vm )
{
    Item* i_port = vm->param( 0 );

    if ( i_port && !i_port->isInteger() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                              .extra( "[I]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    if ( i_port ) // set port
    {
        conn->hostPort( 0, i_port->asInteger() );
        vm->retval( self );
    }
    else // get port
    {
        vm->retval( conn->port() );
    }
}

/*#
    @method connect MongoDB
    @brief Connect or reconnect to MongoDB server.
 */
FALCON_FUNC MongoDBConnection_connect( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    int ret = conn->connect();
    if ( ret != mongo_conn_success )
    {
        switch ( ret )
        {
        case -1:
            throw new MongoDBError( ErrorParam( MONGODB_ERR_NOMEM, __LINE__ )
                                    .desc( FAL_STR( _err_nomem ) ) );
        case mongo_conn_bad_arg:
            throw new MongoDBError( ErrorParam( MONGODB_ERR_CONNECT, __LINE__ )
                                    .desc( FAL_STR( _err_connect_bad_arg ) ) );
        case mongo_conn_no_socket:
            throw new MongoDBError( ErrorParam( MONGODB_ERR_CONNECT, __LINE__ )
                                    .desc( FAL_STR( _err_connect_no_socket ) ) );
        case mongo_conn_not_master:
            throw new MongoDBError( ErrorParam( MONGODB_ERR_CONNECT, __LINE__ )
                                    .desc( FAL_STR( _err_connect_not_master ) ) );
        case mongo_conn_fail:
        default:
            throw new MongoDBError( ErrorParam( MONGODB_ERR_CONNECT, __LINE__ )
                                    .desc( FAL_STR( _err_connect_fail ) ) );
        }
    }
}


/*#
    @method disconnect MongoDB
    @brief Disconnect to MongoDB server.
 */
FALCON_FUNC MongoDBConnection_disconnect( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    int ret = conn->disconnect();
    // we just ignore errors.
    if ( ret ) printf( "Disconnection error (%d)\n", ret );
}


/*#
    @method isConnected MongoDB
    @return true if connected
 */
FALCON_FUNC MongoDBConnection_isConnected( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    vm->retval( conn->isConnected() );
}


/*#
    @method authenticate MongoDB
    @param db
    @param user
    @param pass
    @return true if authenticated
 */
FALCON_FUNC MongoDBConnection_authenticate( VMachine* vm )
{
    Item* i_db = vm->param( 0 );
    Item* i_user = vm->param( 1 );
    Item* i_pass = vm->param( 2 );

    if ( !i_db || !i_db->isString()
        || !i_user || !i_user->isString()
        || !i_pass || !i_pass->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,S,S" ) );
    }

    AutoCString zDB( *i_db );
    AutoCString zUser( *i_user );
    AutoCString zPass( *i_pass );

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    bool b = conn->authenticate( zDB.c_str(), zUser.c_str(), zPass.c_str() );
    vm->retval( b );
}


/*#
    @method addUser MongoDB
    @param db
    @param user
    @param pass
    @return true if user was added
 */
FALCON_FUNC MongoDBConnection_addUser( VMachine* vm )
{
    Item* i_db = vm->param( 0 );
    Item* i_user = vm->param( 1 );
    Item* i_pass = vm->param( 2 );

    if ( !i_db || !i_db->isString()
        || !i_user || !i_user->isString()
        || !i_pass || !i_pass->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,S,S" ) );
    }

    AutoCString zDB( *i_db );
    AutoCString zUser( *i_user );
    AutoCString zPass( *i_pass );

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    bool b = conn->addUser( zDB.c_str(), zUser.c_str(), zPass.c_str() );
    vm->retval( b );
}


/*#
    @method dropDatabase MongoDB
    @param db
    @return true on success
 */
FALCON_FUNC MongoDBConnection_dropDatabase( VMachine* vm )
{
    Item* i_db = vm->param( 0 );

    if ( !i_db || !i_db->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S" ) );
    }

    AutoCString zDB( *i_db );
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    bool b = conn->dropDatabase( zDB.c_str() );
    vm->retval( b );
}


} /* !namespace Ext */
} /* !namespace Falcon */

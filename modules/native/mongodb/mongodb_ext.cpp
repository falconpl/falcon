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

/*******************************************************************************
    MongoDB class
*******************************************************************************/

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

    MongoDB::Connection* conn = 0;
    conn = new MongoDB::Connection( host, port, 0 );
    if ( !conn )
    {
        throw new MongoDBError( ErrorParam( MONGODB_ERR_CREATE_CONN, __LINE__ )
                                .desc( FAL_STR( _err_create_conn ) ) );
    }
    CoreObject* self = vm->self().asObjectSafe();
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
    //if ( ret ) printf( "Disconnection error (%d)\n", ret );
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


/*#
    @method dropCollection MongoDB
    @param db
    @param coll
    @return true on success
 */
FALCON_FUNC MongoDBConnection_dropCollection( VMachine* vm )
{
    Item* i_db = vm->param( 0 );
    Item* i_coll = vm->param( 1 );

    if ( !i_db || !i_db->isString()
        || !i_coll || !i_coll->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,S" ) );
    }

    AutoCString zDB( *i_db );
    AutoCString zColl( *i_coll );
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    bool b = conn->dropCollection( zDB.c_str(), zColl.c_str() );
    vm->retval( b );
}


/*#
    @method insert MongoDB
    @param ns namespace
    @param bson BSONObj instance, or an array of BSON instances
    @return true on success
 */
FALCON_FUNC MongoDBConnection_insert( VMachine* vm )
{
    Item* i_ns = vm->param( 0 );
    Item* i_bobj = vm->param( 1 );

    if ( !i_ns || !i_ns->isString()
        || !i_bobj
        || !( i_bobj->isArray()
        || ( i_bobj->isObject() && i_bobj->asObjectSafe()->derivedFrom( "BSON" ) ) ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,BSON|A" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );
    bool b;

    if ( i_bobj->isObject() )
    {
        CoreObject* obj = i_bobj->asObjectSafe();
        MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( obj->getUserData() );
        b = conn->insert( *i_ns->asString(), bobj );
    }
    else // is array
    {
        AutoCString zNs( *i_ns );
        b = conn->insert( zNs.c_str(), *i_bobj->asArray() );
    }
    vm->retval( b );
}


/*#
    @method update MongoDB
    @param ns namespace
    @param cond BSON instance (conditions)
    @param op BSON instance (operations)
    @optparam upsert (boolean) default true
    @optparam multiple (boolean) default true
    @return true on success
 */
FALCON_FUNC MongoDBConnection_update( VMachine* vm )
{
    Item* i_ns = vm->param( 0 );
    Item* i_cond = vm->param( 1 );
    Item* i_op = vm->param( 2 );
    Item* i_upsert = vm->param( 3 );
    Item* i_multi = vm->param( 4 );

    if ( !i_ns || !i_ns->isString()
        || !i_cond || !( i_cond->isObject() && i_cond->asObjectSafe()->derivedFrom( "BSON" ) )
        || !i_op || !( i_op->isObject() && i_op->asObjectSafe()->derivedFrom( "BSON" ) )
        || ( i_upsert && !i_upsert->isBoolean() )
        || ( i_multi && !i_multi->isBoolean() ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,BSON,BSON" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    AutoCString zNs( *i_ns );
    MongoDB::BSONObj* cond = static_cast<MongoDB::BSONObj*>( i_cond->asObjectSafe()->getUserData() );
    MongoDB::BSONObj* op = static_cast<MongoDB::BSONObj*>( i_op->asObjectSafe()->getUserData() );
    const bool upsert = i_upsert? i_upsert->asBoolean() : true;
    const bool multi = i_multi ? i_multi->asBoolean() : true;

    vm->retval( conn->update( zNs.c_str(), cond, op, upsert, multi ) );
}


/*#
    @method remove MongoDB
    @param ns namespace
    @param cond BSON instance (conditions)
    @return true on success
 */
FALCON_FUNC MongoDBConnection_remove( VMachine* vm )
{
    Item* i_ns = vm->param( 0 );
    Item* i_cond = vm->param( 1 );

    if ( !i_ns || !i_ns->isString()
        || !i_cond || !( i_cond->isObject() && i_cond->asObjectSafe()->derivedFrom( "BSON" ) ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,BSON" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    AutoCString zNs( *i_ns );
    MongoDB::BSONObj* cond = static_cast<MongoDB::BSONObj*>( i_cond->asObjectSafe()->getUserData() );

    vm->retval( conn->remove( zNs.c_str(), cond ) );
}


/*#
    @method findOne MongoDB
    @param ns namespace
    @optparam query BSON instance
    @return BSON result or nil
 */
FALCON_FUNC MongoDBConnection_findOne( VMachine* vm )
{
    Item* i_ns = vm->param( 0 );
    Item* i_query = vm->param( 1 );

    if ( !i_ns || !i_ns->isString()
        || ( i_query && !( i_query->isObject() && i_query->asObjectSafe()->derivedFrom( "BSON" ) ) ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,[BSON]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    AutoCString zNs( *i_ns );
    MongoDB::BSONObj* ret = 0;
    bool b;

    if ( i_query )
    {
        MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( i_query->asObjectSafe()->getUserData() );
        b = conn->findOne( zNs.c_str(), bobj, &ret );
    }
    else
        b = conn->findOne( zNs.c_str(), 0, &ret );

    if ( b )
    {
        fassert( ret );
        Item* wki = vm->findWKI( "BSON" );
        CoreObject* obj = wki->asClass()->createInstance();
        fassert( !obj->getUserData() );
        obj->setUserData( ret );
        vm->retval( obj );
    }
    else
        vm->retnil();
}


/*#
    @method find MongoDB
    @optparam query BSON instance
    @optparam fields BSON instance
    @optparam skip default 0
    @optparam limit default 0 (all)
    @return An array of BSON results or nil
 */
FALCON_FUNC MongoDBConnection_find( VMachine* vm )
{
    Item* i_ns = vm->param( 0 );
    Item* i_query = vm->param( 1 );
    Item* i_fields = vm->param( 2 );
    Item* i_skip = vm->param( 3 );
    Item* i_limit = vm->param( 4 );

    if ( !i_ns || !i_ns->isString()
        || ( i_query && !( i_query->isObject() && i_query->asObjectSafe()->derivedFrom( "BSON" ) ) )
        || ( i_fields && !( i_fields->isObject() && i_fields->asObjectSafe()->derivedFrom( "BSON" ) ) )
        || ( i_skip && !i_skip->isInteger() )
        || ( i_limit && !i_limit->isInteger() ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,[BSON,BSON,I,I]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    AutoCString zNs( *i_ns );
    MongoDB::BSONObj* query = i_query ?
        static_cast<MongoDB::BSONObj*>( i_query->asObjectSafe()->getUserData() ) : 0;
    MongoDB::BSONObj* fields = i_fields ?
        static_cast<MongoDB::BSONObj*>( i_fields->asObjectSafe()->getUserData() ) : 0;
    const int skip = i_skip ? i_skip->asInteger() : 0;
    const int limit = i_limit ? i_limit->asInteger() : 0;
    CoreArray* res;

    const bool b = conn->find( zNs.c_str(), query, fields, skip, limit, &res );
    if ( b )
        vm->retval( res );
    else
        vm->retnil();
}


/*#
    @method count MongoDB
    @param db
    @param coll
    @optparam query BSON instance
    @return Total count or -1 on error
 */
FALCON_FUNC MongoDBConnection_count( VMachine* vm )
{
    Item* i_db = vm->param( 0 );
    Item* i_coll = vm->param( 1 );
    Item* i_query = vm->param( 2 );

    if ( !i_db || !i_db->isString()
        || !i_coll || !i_coll->isString()
        || ( i_query && !( i_query->isObject() && i_query->asObjectSafe()->derivedFrom( "BSON" ) ) ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,S,[BSON]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::Connection* conn = static_cast<MongoDB::Connection*>( self->getUserData() );

    AutoCString db( *i_db );
    AutoCString coll( *i_coll );
    int64 n = -1;

    if ( i_query )
    {
        MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( i_query->asObjectSafe()->getUserData() );
        n = conn->count( db.c_str(), coll.c_str(), bobj );
    }
    else
        n = conn->count( db.c_str(), coll.c_str() );

    vm->retval( n );
}

/*******************************************************************************
    ObjectID class
*******************************************************************************/

/*#
    @class ObjectID
    @brief Mongo Object ID
    @optparam string A string representing an object Id.
 */
FALCON_FUNC MongoOID_init( VMachine* vm )
{
    Item* i_s = vm->param( 0 );

    if ( i_s && !i_s->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "[S]" ) );
    }

    MongoDB::ObjectID* self = static_cast<MongoDB::ObjectID*>( vm->self().asObjectSafe() );

    if ( i_s )
    {
        AutoCString zStr( *i_s );
        self->fromString( zStr.c_str() );
    }

    vm->retval( self );
}


/*#
    @method toString ObjectID
 */
FALCON_FUNC MongoOID_toString( VMachine* vm )
{
    MongoDB::ObjectID* self = static_cast<MongoDB::ObjectID*>( vm->self().asObjectSafe() );
    String s( self->toString() );
    s.bufferize();
    vm->retval( s );
}


/*******************************************************************************
    BSON class
*******************************************************************************/

/*#
    @class BSON
    @brief Create a BSON object.
    @optparam param An integer (reserved space for internal buffer) or a dict to append.

    If no dict is given, an "empty" bson object is created.
 */
FALCON_FUNC MongoBSON_init( VMachine* vm )
{
    Item* i_parm = vm->param( 0 );

    if ( i_parm && !( i_parm->isInteger() || i_parm->isDict() ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "[I|D]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    const int bytes = i_parm && i_parm->isInteger() ? i_parm->asInteger() : 0;
    MongoDB::BSONObj* bobj = 0;
    bobj = new MongoDB::BSONObj( bytes );

    if ( !bobj )
    {
        throw new MongoDBError( ErrorParam( MONGODB_ERR_CREATE_BSON, __LINE__ )
                                .desc( FAL_STR( _err_create_bsonobj ) ) );
    }

    if ( i_parm && i_parm->isDict() ) // append the data
    {
        const int ret = bobj->appendMany( *i_parm->asDict() );
        if ( ret == 1 ) // bad key
        {
            delete bobj;
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( "S" ) );
        }
        else
        if ( ret == 2 ) // bad value
        {
            delete bobj;
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( FAL_STR( _err_inv_item ) ) );
        }
    }

    self->setUserData( bobj );
    vm->retval( self );
}

/*#
    @method reset BSON
    @optparam bytes Reserve some space for internal buffer.
    @brief Clear the BSON object, making it an "empty" one.
 */
FALCON_FUNC MongoBSON_reset( VMachine* vm )
{
    Item* i_bytes = vm->param( 0 );

    if ( i_bytes && !i_bytes->isInteger() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "[I]" ) );
    }

    const int bytes = i_bytes ? i_bytes->asInteger() : 0;
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );
    bobj->reset( bytes );
}


/*#
    @method genOID BSON
    @optparam name Key name (default "_id")
    @brief Generate and append an OID.
    @return self
 */
FALCON_FUNC MongoBSON_genOID( VMachine* vm )
{
    Item* i_nm = vm->param( 0 );

    if ( i_nm && !i_nm->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "[S]" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );

    if ( i_nm )
    {
        AutoCString zNm( *i_nm );
        bobj->genOID( zNm.c_str() );
    }
    else
        bobj->genOID( "_id" );

    vm->retval( self );
}


/*#
    @method append BSON
    @param dict A dict (with keys that must be strings...)
    @brief Append some data to the BSON object
    @return self

    Example:
    @code
        import from mongo
        obj = mongo.BSON().genOID().append( [ "key" => "value" ] )
    @endcode
 */
FALCON_FUNC MongoBSON_append( VMachine* vm )
{
    Item* i_dic = vm->param( 0 );

    if ( !i_dic || !i_dic->isDict() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "D" ) );
    }

    CoreDict* dic = i_dic->asDict();
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );
    const int ret = bobj->appendMany( *dic );
    if ( ret == 1 ) // bad key
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                    .extra( "S" ) );
    }
    else
    if ( ret == 2 ) // bad value
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                    .extra( FAL_STR( _err_inv_item ) ) );
    }
    vm->retval( self );
}


/*#
    @method asDict BSON
    @brief Return a dict representing the BSON object.
 */
FALCON_FUNC MongoBSON_asDict( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );
    vm->retval( bobj->asDict() );
}


/*#
    @method hasKey BSON
    @param key
    @return true if BSON has that key
 */
FALCON_FUNC MongoBSON_hasKey( VMachine* vm )
{
    Item* i_key = vm->param( 0 );

    if ( !i_key || !i_key->isString() )
    {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );
    AutoCString key( *i_key );
    vm->retval( bobj->hasKey( key.c_str() ) );
}


/*#
    @method value BSON
    @param key
    @return value for key given (might be nil), or nil.
 */
FALCON_FUNC MongoBSON_value( VMachine* vm )
{
    Item* i_key = vm->param( 0 );

    if ( !i_key || !i_key->isString() )
    {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( self->getUserData() );
    AutoCString key( *i_key );
    Item* it = bobj->value( key.c_str() );
    if ( it )
        vm->retval( *it );
    else
        vm->retnil();
}

/*******************************************************************************
    BSONIter class
*******************************************************************************/

/*#
    @class BSONIter
    @brief Iterator for BSON objects
    @param bson A BSON object

    Example:
    @code
        iter = BSONIter( bson )
        while iter.next()
            doSomething( iter.key(), iter.value() )
        end
    @endcode

    The iterator copies data from given BSON object, and is completely
    independant and cannot be changed or updated. (This is of course suboptimal
    and could be optimized later.)
 */
FALCON_FUNC MongoBSONIter_init( VMachine* vm )
{
    Item* i_data = vm->param( 0 );

    if ( !i_data
        || !( i_data->isObject() && i_data->asObjectSafe()->derivedFrom( "BSON" ) ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "BSON" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    CoreObject* data = i_data->asObjectSafe();
    MongoDB::BSONObj* bobj = static_cast<MongoDB::BSONObj*>( data->getUserData() );
    MongoDB::BSONIter* iter = new MongoDB::BSONIter( bobj );
    self->setUserData( iter );
    vm->retval( self );
}


/*#
    @method next BSONIter
    @brief Return true if there is more data to iterate over
 */
FALCON_FUNC MongoBSONIter_next( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONIter* iter = static_cast<MongoDB::BSONIter*>( self->getUserData() );
    vm->retval( iter->next() );
}


/*#
    @method key BSONIter
    @brief Get the current BSON key string
 */
FALCON_FUNC MongoBSONIter_key( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONIter* iter = static_cast<MongoDB::BSONIter*>( self->getUserData() );
    const char* k = iter->currentKey();
    if ( k )
    {
        String s( k );
        s.bufferize();
        vm->retval( s );
    }
    else
        vm->retnil();
}


/*#
    @method value BSONIter
    @brief Get the current BSON value
 */
FALCON_FUNC MongoBSONIter_value( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONIter* iter = static_cast<MongoDB::BSONIter*>( self->getUserData() );
    Item* v = iter->currentValue();
    if ( v )
        vm->retval( *v );
    else
        vm->retnil();
}


/*#
    @method reset BSONIter
    @brief Reset to start of iterator.
 */
FALCON_FUNC MongoBSONIter_reset( VMachine* vm )
{
    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONIter* iter = static_cast<MongoDB::BSONIter*>( self->getUserData() );
    iter->reset();
}


/*#
    @method find BSONIter
    @param name Key name
    @brief Return true when (and set iterator position where) name is found in the BSON.

    If false is returned, iterator is at end (and you may have to reset it).
    This method does a reset before searching.
 */
FALCON_FUNC MongoBSONIter_find( VMachine* vm )
{
    Item* i_nm = vm->param( 0 );

    if ( !i_nm || !i_nm->isString() )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S" ) );
    }

    CoreObject* self = vm->self().asObjectSafe();
    MongoDB::BSONIter* iter = static_cast<MongoDB::BSONIter*>( self->getUserData() );
    AutoCString zNm( *i_nm->asString() );
    vm->retval( iter->find( zNm.c_str() ) );
}


} /* !namespace Ext */
} /* !namespace Falcon */

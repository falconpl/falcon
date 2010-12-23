/*
 *  Falcon MongoDB - Module definition
 */

#include "mongodb_ext.h"
#include "mongodb_mod.h"
#include "mongodb_srv.h"
#include "version.h"

#include <falcon/module.h>

/*#
   @module mongodb MongoDB driver module
   @brief Client module for the MongoDB database ( http://www.mongodb.org/ )
*/

/*
    TODO LIST...

    - handling error better
    - Optimize BSONIter so that they dont copy data (?)
    - think about what to put in the service
    - JSON interactions
 */

Falcon::MongoDBService theMongoDBService;


FALCON_MODULE_DECL
{
    Falcon::Module *self = new Falcon::Module();
    self->name( "mongo" );
    self->engineVersion( FALCON_VERSION_NUM );
    self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

    #include "mongodb_st.h"

    // MongoDB class
    Falcon::Symbol *dbconn_cls = self->addClass( "MongoDB",
                                                 Falcon::Ext::MongoDBConnection_init );
    dbconn_cls->setWKS( true );

    self->addClassMethod( dbconn_cls, "host",
                          Falcon::Ext::MongoDBConnection_host );
    self->addClassMethod( dbconn_cls, "port",
                          Falcon::Ext::MongoDBConnection_port );
    self->addClassMethod( dbconn_cls, "connect",
                          Falcon::Ext::MongoDBConnection_connect );
    self->addClassMethod( dbconn_cls, "disconnect",
                          Falcon::Ext::MongoDBConnection_disconnect );
    self->addClassMethod( dbconn_cls, "isConnected",
                          Falcon::Ext::MongoDBConnection_isConnected );
    self->addClassMethod( dbconn_cls, "authenticate",
                          Falcon::Ext::MongoDBConnection_authenticate );
    self->addClassMethod( dbconn_cls, "addUser",
                          Falcon::Ext::MongoDBConnection_addUser );
    self->addClassMethod( dbconn_cls, "dropDatabase",
                          Falcon::Ext::MongoDBConnection_dropDatabase );
    self->addClassMethod( dbconn_cls, "dropCollection",
                          Falcon::Ext::MongoDBConnection_dropCollection );
    self->addClassMethod( dbconn_cls, "insert",
                          Falcon::Ext::MongoDBConnection_insert );
    self->addClassMethod( dbconn_cls, "findOne",
                          Falcon::Ext::MongoDBConnection_findOne );
    self->addClassMethod( dbconn_cls, "count",
                          Falcon::Ext::MongoDBConnection_count );

    // ObjectID class
    Falcon::Symbol *oid_cls = self->addClass( "ObjectID",
                                              Falcon::Ext::MongoOID_init );
    oid_cls->setWKS( true );
    oid_cls->getClassDef()->factory( Falcon::MongoDB::ObjectID::factory );

    self->addClassMethod( oid_cls, "toString",
                          Falcon::Ext::MongoOID_toString );

    // BSON class
    Falcon::Symbol* bson_cls = self->addClass( "BSON",
                                               Falcon::Ext::MongoBSON_init );
    bson_cls->setWKS( true );
    self->addClassMethod( bson_cls, "reset",
                          Falcon::Ext::MongoBSON_reset );
    self->addClassMethod( bson_cls, "genOID",
                          Falcon::Ext::MongoBSON_genOID );
    self->addClassMethod( bson_cls, "append",
                          Falcon::Ext::MongoBSON_append );
    self->addClassMethod( bson_cls, "asDict",
                          Falcon::Ext::MongoBSON_asDict );

    // BSONIter class
    Falcon::Symbol* bsonit_cls = self->addClass( "BSONIter",
                                                 Falcon::Ext::MongoBSONIter_init );
    bsonit_cls->setWKS( true );
    self->addClassMethod( bsonit_cls, "next",
                          Falcon::Ext::MongoBSONIter_next );
    self->addClassMethod( bsonit_cls, "key",
                          Falcon::Ext::MongoBSONIter_key );
    self->addClassMethod( bsonit_cls, "value",
                          Falcon::Ext::MongoBSONIter_value );
    self->addClassMethod( bsonit_cls, "reset",
                          Falcon::Ext::MongoBSONIter_reset );
    self->addClassMethod( bsonit_cls, "find",
                          Falcon::Ext::MongoBSONIter_find );

    // MongoDBError class
    Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
    Falcon::Symbol *err_cls = self->addClass( "MongoDBError", &Falcon::Ext::MongoDBError_init );
    err_cls->setWKS( true );
    err_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

    // service publication
    self->publishService( &theMongoDBService );

    return self;
}

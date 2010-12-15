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

    // MongoDBError class
    Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
    Falcon::Symbol *err_cls = self->addClass( "MongoDBError", &Falcon::Ext::MongoDBError_init );
    err_cls->setWKS( true );
    err_cls->getClassDef()->addInheritance( new Falcon::InheritDef( error_class ) );

    // service publication
    self->publishService( &theMongoDBService );

    return self;
}

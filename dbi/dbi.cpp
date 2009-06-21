/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi.cpp
 *
 * Database common interface.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun Dec 2007 23 21:54:34 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include "dbi.h"
#include "version.h"
#include "dbi_ext.h"

/*#
 @module dbi The DBI Falcon Module.
 @brief Main module for the Falcon DBI module suite.

 This is the base of the Falcon DBI subsystem.
 This DBI module relies optionally on several database access libraries including:

 - <a target="_new" href="http://postgresql.org/">PostgreSQL</a>
 - <a target="_new" href="http://mysql.com/">MySQL</a>
 - <a target="_new" href="http://sqlite.org/">SQLite</a>
 - <a target="_new" href="http://it.wikipedia.org/wiki/ODBC">ODBC</a>
 
 One or more database libraries are required to make DBI useful.

 @beginmodule dbi
*/

// Instantiate the loader service
Falcon::DBILoaderImpl theDBIService;

// the main module
FALCON_MODULE_DECL
{
   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "dbi" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // main factory function
   self->addExtFunc( "DBIConnect", Falcon::Ext::DBIConnect )->
      addParam("String");

   /*#
      @class DBIBaseTrans
      @brief Base class for DBI querable and updatable items.

      This class is the base for database handles and single transactions (dabatase
      handle portions that some database engine is able to open separately).

      The vast majority of common query operations available in databases are
      exposed by this class, which is inhertited by @a DBIHandle and @a DBITransaction.
   */
   Falcon::Symbol *btrans_class = self->addClass( "%DBIBaseTrans", false ); // private class
   btrans_class->setWKS( true );
   self->addClassMethod( btrans_class, "query",          Falcon::Ext::DBIBaseTrans_query );
   self->addClassMethod( btrans_class, "queryOne",          Falcon::Ext::DBIBaseTrans_queryOne );
   self->addClassMethod( btrans_class, "queryOneArray",     Falcon::Ext::DBIBaseTrans_queryOneArray );
   self->addClassMethod( btrans_class, "queryOneDict",      Falcon::Ext::DBIBaseTrans_queryOneDict );
   self->addClassMethod( btrans_class, "queryOneObject",    Falcon::Ext::DBIBaseTrans_queryOneObject );
   self->addClassMethod( btrans_class, "commit",            Falcon::Ext::DBIBaseTrans_commit );
   self->addClassMethod( btrans_class, "rollback",          Falcon::Ext::DBIBaseTrans_rollback );
   self->addClassMethod( btrans_class, "getLastError",      Falcon::Ext::DBIBaseTrans_getLastError );
   
   self->addClassMethod( btrans_class, "insert",            Falcon::Ext::DBIBaseTrans_insert );
   self->addClassMethod( btrans_class, "update",            Falcon::Ext::DBIBaseTrans_update );
   self->addClassMethod( btrans_class, "delete",            Falcon::Ext::DBIBaseTrans_delete );

   /*#
    @class DBIHandle
    @brief DBI connection handle returned by @a DBIConnect.

    You will not instantiate this class directly, instead, you must use @a DBIConnect.
    */

   // create the base class DBIHandler for falcon
   Falcon::Symbol *handler_class = self->addClass( "%DBIHandle" ); // private class
   handler_class->getClassDef()->addInheritance( new Falcon::InheritDef(btrans_class) );
   handler_class->setWKS( true );
   self->addClassMethod( handler_class, "startTransaction",  Falcon::Ext::DBIHandle_startTransaction );
   self->addClassMethod( handler_class, "sqlExpand",         Falcon::Ext::DBIHandle_sqlExpand );
   self->addClassMethod( handler_class, "getLastInsertedId", Falcon::Ext::DBIHandle_getLastInsertedId );
   self->addClassMethod( handler_class, "close",             Falcon::Ext::DBIHandle_close );


   /*#
    @class DBITransaction
    @brief Represents one transaction in the underlying database server.

    You will not instantiate this class directly, instead, you must use
    the startTransaction method of your @a DBIHandle.
    */

   // create the base class DBITransaction for falcon
   Falcon::Symbol *trans_class = self->addClass( "%DBITransaction", false ); // private class
   trans_class->getClassDef()->addInheritance( new Falcon::InheritDef(btrans_class) );
   trans_class->setWKS( true );
   self->addClassMethod( trans_class, "openBlob",    Falcon::Ext::DBITransaction_openBlob );
   self->addClassMethod( trans_class, "createBlob",  Falcon::Ext::DBITransaction_createBlob );
   self->addClassMethod( trans_class, "readBlob",    Falcon::Ext::DBITransaction_readBlob );
   self->addClassMethod( trans_class, "writeBlob",   Falcon::Ext::DBITransaction_writeBlob );
   self->addClassMethod( trans_class, "close",       Falcon::Ext::DBITransaction_close );


   /*#
    @class DBIRecordset
    @brief Represent a collection of database records as required from @a DBIBaseTrans.query.
    You will not instantiate this class directly, instead, you must use @a DBIBaseTrans.query.
    */

   // create the base class DBIRecordset for falcon
   Falcon::Symbol *rs_class = self->addClass( "%DBIRecordset", false ); // private class
   rs_class->setWKS( true );
   self->addClassMethod( rs_class, "next",           Falcon::Ext::DBIRecordset_next );
   self->addClassMethod( rs_class, "fetchArray",     Falcon::Ext::DBIRecordset_fetchArray );
   self->addClassMethod( rs_class, "fetchDict",      Falcon::Ext::DBIRecordset_fetchDict );
   self->addClassMethod( rs_class, "fetchObject",    Falcon::Ext::DBIRecordset_fetchObject );
   self->addClassMethod( rs_class, "asString",       Falcon::Ext::DBIRecordset_asString );
   self->addClassMethod( rs_class, "asBoolean",      Falcon::Ext::DBIRecordset_asBoolean );
   self->addClassMethod( rs_class, "asInteger",      Falcon::Ext::DBIRecordset_asInteger );
   self->addClassMethod( rs_class, "asInteger64",    Falcon::Ext::DBIRecordset_asInteger64 );
   self->addClassMethod( rs_class, "asNumeric",      Falcon::Ext::DBIRecordset_asNumeric );
   self->addClassMethod( rs_class, "asDate",         Falcon::Ext::DBIRecordset_asDate );
   self->addClassMethod( rs_class, "asTime",         Falcon::Ext::DBIRecordset_asTime );
   self->addClassMethod( rs_class, "asDateTime",     Falcon::Ext::DBIRecordset_asDateTime );
   self->addClassMethod( rs_class, "asBlobID",       Falcon::Ext::DBIRecordset_asBlobID );
   self->addClassMethod( rs_class, "getRowCount",    Falcon::Ext::DBIRecordset_getRowCount );
   self->addClassMethod( rs_class, "getColumnCount", Falcon::Ext::DBIRecordset_getColumnCount );
   self->addClassMethod( rs_class, "getColumnTypes", Falcon::Ext::DBIRecordset_getColumnTypes );
   self->addClassMethod( rs_class, "getColumnNames", Falcon::Ext::DBIRecordset_getColumnNames );
   self->addClassMethod( rs_class, "getLastError",   Falcon::Ext::DBIRecordset_getLastError );
   self->addClassMethod( rs_class, "close",          Falcon::Ext::DBIRecordset_close );

   /*#
    @class DBIRecord
    @brief Base class for object oriented database access.

    @prop _dbh database handle used for this instance
    @prop _tableName database table name this instance should read from and write to
    @prop _primaryKey primary key used during get and update data
    @prop _persist an optional array of class properties to read from and write to
    the database.

    @section Persistence Rules

    All properties begining with an underscore (_) will be ignored. All other
    properties will be persisted during a @a DBIRecord.insert or @a DBIRecord.update
    unless the property _persist is defined. If _persist
    is defined, then all properties will be ignored except those in the
    _persist array.

    In the below example, the class Person, does not need to define the
    _persist property as it has no special attributes. It was included
    as an example of how to define the _persist property.

    @section Example

    @code
    load dbi

    class Person( nName, nDob ) from DBIRecord
       _tableName = "names"
       _primaryKey = "name"
       _persist = ["name", "dob"]

       name = nName
       dob = nDob
    end

    db = DBIConnect( "sqlite3:example.db" )
    Person( "John Doe", TimeStamp() ).insert()
    Person( "Jane Doe", TimeStamp() ).insert()

    r = db.query( "SELECT * FROM names" )
    while (n = r.fetchObject( Person() ))
      > n.name, " was born on ", n.dob
      n.name = n.name + "ey"
      n.update()
    end
    r.close()
    db.close()
    @endcode

    You can of course expand the Person class with more properties and also methods
    which are not persisted, such as a method to calculate the age of the person,
    or to determine if today is their birthday which in turn would show the real
    power of using the DBIRecord object method.
    */

   // create the base class DBIRecord for falcon
   Falcon::Symbol *rec_class = self->addClass( "DBIRecord", Falcon::Ext::DBIRecord_init );
   self->addClassMethod(   rec_class, "insert",      Falcon::Ext::DBIRecord_insert );
   self->addClassMethod(   rec_class, "update",      Falcon::Ext::DBIRecord_update );
   self->addClassMethod(   rec_class, "delete",      Falcon::Ext::DBIRecord_delete );
   self->addClassProperty( rec_class, "_dbh" );
   self->addClassProperty( rec_class, "_tableName" );
   self->addClassProperty( rec_class, "_primaryKey" );
   self->addClassProperty( rec_class, "_persist" );
   // TODO: actually use this
   //self->addGlobal( "DBIRecord__defaultDBH" );  // Static class variable

   // service publication
   self->publishService( &theDBIService );

   /*#
    @class DBIError
    @brief DBI specific error.

    Inherited class from Error to distinguish from a standard Falcon error. In many
    cases, DBIError.extra will contain the SQL query that caused the problem.
    */

   // create the base class DBIError for falcon
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DBIError", Falcon::Ext::DBIError_init );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   /*#
    @class DBIBlobStream
    @brief Specific DBI stream used to read and write to and from blobs.

    Inherits from the Stream class in the RTL. This is actually an abstract class;
    instances of it can be returned by some blob-related methods in the DBI class.
   */

   Falcon::Symbol *stream_class = self->addExternalRef( "Stream" ); // it's external
   Falcon::Symbol *blobstream_cls = self->addClass( "DBIBlobStream" );
   blobstream_cls->setWKS( true ); // can be created by the VM but...
   blobstream_cls->exported( false ); // ... not instantiable by the scripts
   blobstream_cls->getClassDef()->addInheritance( new Falcon::InheritDef( stream_class ) );
   self->addClassMethod( blobstream_cls, "getBlobID", Falcon::Ext::DBIBlobStream_getBlobID );

   // we're done
   return self;
}

/* end of dbi.cpp */



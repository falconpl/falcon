/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi.cpp
 *
 * Short description
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Sun Dec 2007 23 21:54:34 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include "dbi.h"
#include "version.h"
#include "dbi_ext.h"

// Instantiate the loader service
Falcon::DBILoaderImpl theDBIService;

// the main module
FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "dbi" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   // main factory function
   self->addExtFunc( "DBIConnect", Falcon::Ext::DBIConnect );

   // create the base class DBIHandler for falcon
   Falcon::Symbol *handler_class = self->addClass( "%DBIHandler" ); // private class
   self->addClassMethod( handler_class, "startTransaction",  Falcon::Ext::DBIHandle_startTransaction );
   self->addClassMethod( handler_class, "query",             Falcon::Ext::DBIHandle_query );
   self->addClassMethod( handler_class, "queryOne",          Falcon::Ext::DBIHandle_queryOne );
   self->addClassMethod( handler_class, "queryOneArray",     Falcon::Ext::DBIHandle_queryOneArray );
   self->addClassMethod( handler_class, "queryOneDict",      Falcon::Ext::DBIHandle_queryOneDict );
   self->addClassMethod( handler_class, "queryOneObject",    Falcon::Ext::DBIHandle_queryOneObject );
   self->addClassMethod( handler_class, "execute",           Falcon::Ext::DBIHandle_execute );
   self->addClassMethod( handler_class, "sqlExpand",         Falcon::Ext::DBIHandle_sqlExpand );
   self->addClassMethod( handler_class, "getLastInsertedId", Falcon::Ext::DBIHandle_getLastInsertedId );
   self->addClassMethod( handler_class, "getLastError",      Falcon::Ext::DBIHandle_getLastError );
   self->addClassMethod( handler_class, "close",             Falcon::Ext::DBIHandle_close );

   // create the base class DBITransaction for falcon
   Falcon::Symbol *trans_class = self->addClass( "%DBITransaction" ); // private class
   self->addClassMethod( trans_class, "query",    Falcon::Ext::DBITransaction_query );
   self->addClassMethod( trans_class, "execute",  Falcon::Ext::DBITransaction_execute );
   self->addClassMethod( trans_class, "commit",   Falcon::Ext::DBITransaction_commit );
   self->addClassMethod( trans_class, "rollback", Falcon::Ext::DBITransaction_rollback );
   self->addClassMethod( trans_class, "close",    Falcon::Ext::DBITransaction_close );

   // create the base class DBIRecordset for falcon
   Falcon::Symbol *rs_class = self->addClass( "%DBIRecordset" ); // private class
   self->addClassMethod( rs_class, "next",           Falcon::Ext::DBIRecordset_next );
   self->addClassMethod( rs_class, "fetchArray",     Falcon::Ext::DBIRecordset_fetchArray );
   self->addClassMethod( rs_class, "fetchDict",      Falcon::Ext::DBIRecordset_fetchDict );
   self->addClassMethod( rs_class, "fetchObject",    Falcon::Ext::DBIRecordset_fetchObject );
   self->addClassMethod( rs_class, "asString",       Falcon::Ext::DBIRecordset_asString );
   self->addClassMethod( rs_class, "asInteger",      Falcon::Ext::DBIRecordset_asInteger );
   self->addClassMethod( rs_class, "asInteger64",    Falcon::Ext::DBIRecordset_asInteger64 );
   self->addClassMethod( rs_class, "asNumeric",      Falcon::Ext::DBIRecordset_asNumeric );
   self->addClassMethod( rs_class, "asDate",         Falcon::Ext::DBIRecordset_asDate );
   self->addClassMethod( rs_class, "asTime",         Falcon::Ext::DBIRecordset_asTime );
   self->addClassMethod( rs_class, "asDateTime",     Falcon::Ext::DBIRecordset_asDateTime );
   self->addClassMethod( rs_class, "getRowCount",    Falcon::Ext::DBIRecordset_getRowCount );
   self->addClassMethod( rs_class, "getColumnCount", Falcon::Ext::DBIRecordset_getColumnCount );
   self->addClassMethod( rs_class, "getColumnTypes", Falcon::Ext::DBIRecordset_getColumnTypes );
   self->addClassMethod( rs_class, "getColumnNames", Falcon::Ext::DBIRecordset_getColumnNames );
   self->addClassMethod( rs_class, "getLastError",   Falcon::Ext::DBIRecordset_getLastError );
   self->addClassMethod( rs_class, "close",          Falcon::Ext::DBIRecordset_close );

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
   self->addGlobal( "DBIRecord__defaultDBH" );  // Static class variable

   // service publication
   self->publishService( &theDBIService );

   // create the base class DBIError for falcon
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *procerr_cls = self->addClass( "DBIError", Falcon::Ext::DBIError_init );
   procerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   // we're done
   return self;
}

/* end of dbi.cpp */


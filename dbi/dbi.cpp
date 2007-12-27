/*
   FALCON - The Falcon Programming Language.
   FILE: dbi.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun Dec 2007 23 21:54:34 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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
   self->addClassMethod( handler_class, "startTransaction", Falcon::Ext::DBIHandle_startTransaction );
   self->addClassMethod( handler_class, "query", Falcon::Ext::DBIHandle_query );
   self->addClassMethod( handler_class, "execute", Falcon::Ext::DBIHandle_execute );
   self->addClassMethod( handler_class, "close", Falcon::Ext::DBIHandle_close );

   // create the base class DBITransaction for falcon
   Falcon::Symbol *trans_class = self->addClass( "%DBITransaction" ); // private class
   self->addClassMethod( trans_class, "query", Falcon::Ext::DBITransaction_query );
   self->addClassMethod( trans_class, "execute", Falcon::Ext::DBITransaction_execute );
   self->addClassMethod( trans_class, "close", Falcon::Ext::DBITransaction_close );
   
   // create the base class DBIRecordset for falcon
   Falcon::Symbol *rec_class = self->addClass( "%DBIRecordset" ); // private class
   self->addClassMethod( rec_class, "next", Falcon::Ext::DBIRecordset_next );
   self->addClassMethod( rec_class, "fetchArray", Falcon::Ext::DBIRecordset_fetchArray );
   self->addClassMethod( rec_class, "fetchDict", Falcon::Ext::DBIRecordset_fetchDict );
   self->addClassMethod( rec_class, "asString", Falcon::Ext::DBIRecordset_asString );
   self->addClassMethod( rec_class, "asInteger", Falcon::Ext::DBIRecordset_asInteger );
   self->addClassMethod( rec_class, "asInteger64", Falcon::Ext::DBIRecordset_asInteger64 );
   self->addClassMethod( rec_class, "asNumeric", Falcon::Ext::DBIRecordset_asNumeric );
   self->addClassMethod( rec_class, "asDate", Falcon::Ext::DBIRecordset_asDate );
   self->addClassMethod( rec_class, "asTime", Falcon::Ext::DBIRecordset_asTime );
   self->addClassMethod( rec_class, "asDateTime", Falcon::Ext::DBIRecordset_asDateTime );
   self->addClassMethod( rec_class, "getRowCount", Falcon::Ext::DBIRecordset_getRowCount );
   self->addClassMethod( rec_class, "getColumnCount", Falcon::Ext::DBIRecordset_getColumnCount );
   self->addClassMethod( rec_class, "getColumnTypes", Falcon::Ext::DBIRecordset_getColumnTypes );
   self->addClassMethod( rec_class, "getColumnNames", Falcon::Ext::DBIRecordset_getColumnNames );
   self->addClassMethod( rec_class, "getLastError", Falcon::Ext::DBIRecordset_getLastError );
   self->addClassMethod( rec_class, "close", Falcon::Ext::DBIRecordset_close );
   
   // service pubblication
   self->publishService( &theDBIService );

   // we're done
   return self;
}


/* end of dbi.cpp */


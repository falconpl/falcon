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
#include "dbi_st.h"

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
   #define FALCON_DECLARE_MODULE self

   // Module declaration
   Falcon::Module *self = new Falcon::Module();
   self->name( "dbi" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "dbi_st.h"

   // main factory function
   self->addExtFunc( "DBIConnect", &Falcon::Ext::DBIConnect )->
      addParam("params")->addParam("queryops");


   /*#
      @class Statement
      @brief Base for database operation
   */
   Falcon::Symbol *stmt_class = self->addClass( "%Statement", false ); // private class
   stmt_class->setWKS( true );
   self->addClassMethod( stmt_class, "execute", &Falcon::Ext::Statement_execute );
   self->addClassMethod( stmt_class, "reset", &Falcon::Ext::Statement_reset );
   self->addClassMethod( stmt_class, "close", &Falcon::Ext::Statement_close );

   /*#
    @class Handle
    @brief DBI connection handle returned by @a connect.
    */

   // create the base class DBIHandler for falcon
   Falcon::Symbol *handler_class = self->addClass( "%Handle", true );
   handler_class->setWKS( true );
   self->addClassMethod( handler_class, "options", &Falcon::Ext::Handle_options ).asSymbol()
      ->addParam("options");
   self->addClassMethod( handler_class, "query", &Falcon::Ext::Handle_query ).asSymbol()->
         addParam("sql");
   self->addClassMethod( handler_class, "call", &Falcon::Ext::Handle_call ).asSymbol()->
         addParam("sql");
   self->addClassMethod( handler_class, "perform", &Falcon::Ext::Handle_perform ).asSymbol()->
         addParam("sql");
   self->addClassMethod( handler_class, "prepare", &Falcon::Ext::Handle_prepare ).asSymbol()->
         addParam("sql");
   self->addClassMethod( handler_class, "close", &Falcon::Ext::Handle_close );
   self->addClassMethod( handler_class, "getLastID",  &Falcon::Ext::Handle_getLastID ).asSymbol()
         ->addParam("name");
   self->addClassMethod( handler_class, "begin", &Falcon::Ext::Handle_begin );
   self->addClassMethod( handler_class, "commit", &Falcon::Ext::Handle_commit );
   self->addClassMethod( handler_class, "rollback", &Falcon::Ext::Handle_rollback );

   self->addClassMethod( handler_class, "lselect", &Falcon::Ext::Handle_lselect ).asSymbol()
         ->addParam("sql")->addParam("begin")->addParam("count");

   /*#
    @class Recordset
    @brief Represent a collection of database records as required from @a DBIBaseTrans.query.
    */

   // create the base class DBIRecordset for falcon
   Falcon::Symbol *rs_class = self->addClass( "%Recordset", false ); // private class
   rs_class->setWKS( true );
   self->addClassMethod( rs_class, "discard", &Falcon::Ext::Recordset_discard ).asSymbol()->
         addParam( "count" );
   self->addClassMethod( rs_class, "fetch",&Falcon::Ext::Recordset_fetch ).asSymbol()->
            addParam( "item" )->addParam( "count" );
   self->addClassMethod( rs_class, "do", &Falcon::Ext::Recordset_do ).asSymbol()->
            addParam( "cb" )->addParam( "item" );

   self->addClassMethod( rs_class, "getCurrentRow", &Falcon::Ext::Recordset_getCurrentRow );
   self->addClassMethod( rs_class, "getRowCount", &Falcon::Ext::Recordset_getRowCount );
   self->addClassMethod( rs_class, "getColumnCount", &Falcon::Ext::Recordset_getColumnCount );
   self->addClassMethod( rs_class, "getColumnNames", &Falcon::Ext::Recordset_getColumnNames );
   self->addClassMethod( rs_class, "close", &Falcon::Ext::Recordset_close );

   /*#
    @class DBIError
    @brief DBI specific error.

    Inherited class from Error to distinguish from a standard Falcon error. In many
    cases, DBIError.extra will contain the SQL query that caused the problem.
    */

   // create the base class DBIError for falcon
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *dbierr_cls = self->addClass( "DBIError", &Falcon::Ext::DBIError_init );
   dbierr_cls->setWKS( true );
   dbierr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   // service publication
   self->publishService( &theDBIService );

   // we're done
   return self;
}

/* end of dbi.cpp */



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
   self->addExtFunc( "connect", &Falcon::Ext::DBIConnect )->
      addParam("params")->addParam("trops");



   /*#
      @class Transaction
      @brief Base for database operation
   */
   Falcon::Symbol *trans_class = self->addClass( "%Transaction", false ); // private class
   trans_class->setWKS( true );
   self->addClassMethod( trans_class, "query", &Falcon::Ext::Transaction_query ).asSymbol()->
         addParam("sql");
   self->addClassMethod( trans_class, "call", &Falcon::Ext::Transaction_call ).asSymbol()->
         addParam("sql");
   self->addClassMethod( trans_class, "prepare", &Falcon::Ext::Transaction_prepare ).asSymbol()->
         addParam("sql");
   self->addClassMethod( trans_class, "execute", &Falcon::Ext::Transaction_execute );
   self->addClassMethod( trans_class, "commit", &Falcon::Ext::Transaction_commit );
   self->addClassMethod( trans_class, "rollback", &Falcon::Ext::Transaction_rollback );
   self->addClassMethod( trans_class, "close", &Falcon::Ext::Transaction_close );
   self->addClassMethod( trans_class, "tropen", &Falcon::Ext::Transaction_tropen ).asSymbol()
         ->addParam("options");
   self->addClassMethod( trans_class, "getLastID",          &Falcon::Ext::Transaction_getLastID ).asSymbol()
         ->addParam("name");


   /*#
    @class Handle
    @brief DBI connection handle returned by @a connect.

    You will not instantiate this class directly, instead, you must use @a DBIConnect.
    */

   // create the base class DBIHandler for falcon
   Falcon::Symbol *handler_class = self->addClass( "%Handle", true );
   handler_class->setWKS( true );
   self->addClassMethod( handler_class, "trops", &Falcon::Ext::Handle_trops ).asSymbol()
      ->addParam("options");
   self->addClassMethod( handler_class, "tropen", &Falcon::Ext::Handle_tropen ).asSymbol()
      ->addParam("options");
   self->addClassMethod( handler_class, "close", &Falcon::Ext::Handle_close );


   /*#
    @class Recordset
    @brief Represent a collection of database records as required from @a DBIBaseTrans.query.
    You will not instantiate this class directly, instead, you must use @a DBIBaseTrans.query.
    */

   // create the base class DBIRecordset for falcon
   Falcon::Symbol *rs_class = self->addClass( "%Recordset", false ); // private class
   rs_class->setWKS( true );
   self->addClassMethod( rs_class, "discard", &Falcon::Ext::Recordset_discard ).asSymbol()->
         addParam( "count" );
   self->addClassMethod( rs_class, "fetch",&Falcon::Ext::Recordset_fetch ).asSymbol()->
            addParam( "item" )->addParam( "count" );
   //self->addClassMethod( rs_class, "do", &Falcon::Ext::Recordset_do ).asSymbol()->
   //         addParam( "item" );

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



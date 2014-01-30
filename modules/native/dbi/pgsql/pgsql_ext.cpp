/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_ext.cpp
 *
 * PgSQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar, Stanislas Marquis
 * Begin: Sun Dec 23 21:54:42 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/dbi/pgsql/pgsql_ext.cpp"

#include <falcon/vmcontext.h>
#include <falcon/function.h>
#include "../dbi/dbi.h"

#include "pgsql_ext.h"
#include "pgsql_mod.h"

/*#
    @beginmodule dbi.pgsql
 */

namespace Falcon {
namespace Ext {

/*#
    @class PgSQL
    @brief Direct interface to Postgre SQL database.
    @param connect String containing connection parameters.
    @optparam options String containing options

   The @b connect string is directly passed to the low level postgre driver.
 */


/*#
    @method prepareNamed PgSQL
    @brief Prepares a PgSQL specific "named statement".
    @param name Name for the prepared statement
    @param query The query to prepare
 */
FALCON_DECLARE_FUNCTION( prepareNamed, "name:S,query:S" )
FALCON_DEFINE_FUNCTION_P1( prepareNamed )
{
    Item* i_name = ctx->param( 0 );
    Item* i_query = ctx->param( 1 );

    if ( !i_name || !i_name->isString()
        || !i_query || !i_query->isString() )
    {
        throw paramError(__LINE__,SRC);
    }

    DBIHandlePgSQL* dbh = ctx->tself<DBIHandlePgSQL*>();
    fassert( dbh );

    // names of stored procedures need to be lowercased
    String name = *i_name->asString();
    name.lower();

    DBIStatement* trans = dbh->prepareNamed( name, *i_query->asString() );

    // snippet taken from dbi_ext.h - should be shared?
    DBIModule* dbm = static_cast<DBIModule*>(this->methodOf()->module());
    Class* cls = dbm->statementClass();
    ctx->returnFrame( FALCON_GC_STORE(cls, trans) );

}


ClassPGSQLDBIHandle::ClassPGSQLDBIHandle():
         ClassDriverDBIHandle("PgSql")
{}

ClassPGSQLDBIHandle::~ClassPGSQLDBIHandle()
{}

void* ClassPGSQLDBIHandle::createInstance() const
{
   return new DBIHandlePgSQL(this);
}


} /* namespace Ext */
} /* namespace Falcon */

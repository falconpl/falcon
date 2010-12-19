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

#include <falcon/engine.h>

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
 */

/*#
   @init PgSQL
   @brief Connects to a PgSQL database.

   The @b connect string is directly passed to the low level postgre driver.
 */

FALCON_FUNC PgSQL_init( VMachine* vm )
{
    Item* i_params = vm->param( 0 );
    Item* i_opts = vm->param( 1 );

    if ( !i_params || !i_params->isString()
        || ( i_opts != 0 && !i_opts->isString() ) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                .extra( "S,[S]" ) );
    }

    String* params = i_params->asString();
    DBIHandle* hand = 0;

    try
    {
        hand = thePgSQLService.connect( *params );
        if ( i_opts != 0 )
            hand->options( *i_opts->asString() );
        // great, we have the database handler open. Now we must create a falcon object to store it.
        CoreObject* instance = thePgSQLService.makeInstance( vm, hand );
        vm->retval( instance );
    }
    catch ( DBIError* error )
    {
        delete hand;
        throw error;
    }
}


/*#
    @method prepareNamed PgSQL
    @param name Name for the prepared statement
    @param query The query to prepare
 */
FALCON_FUNC PgSQL_prepareNamed( VMachine* vm )
{
    Item* i_name = vm->param( 0 );
    Item* i_query = vm->param( 1 );

    if ( !i_name || !i_name->isString()
        || !i_query || !i_query->isString() )
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,S" ) );

    DBIHandlePgSQL* dbh = static_cast<DBIHandlePgSQL*>( vm->self().asObjectSafe()->getUserData() );
    fassert( dbh );

    // names of stored procedures need to be lowercased
    String name = *i_name->asString();
    name.lower();

    DBIStatement* trans = dbh->prepareNamed( name, *i_query->asString() );

    // snippet taken from dbi_ext.h - should be shared?
    Item *trclass = vm->findWKI( "%Statement" );
    fassert( trclass != 0 && trclass->isClass() );
    CoreObject *oth = trclass->asClass()->createInstance();
    oth->setUserData( trans );
    vm->retval( oth );
}


} /* namespace Ext */
} /* namespace Falcon */

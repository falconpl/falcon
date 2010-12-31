/*
 * FALCON - The Falcon Programming Language.
 * FILE: oracle_ext.cpp
 *
 * Oracle Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Steven Oliver
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>

#include "oracle_mod.h"
#include "oracle_ext.h"

/*--# << turn on when active

   @beginmodule dbi.oracle
*/
namespace Falcon
{
    namespace Ext
    {

/*--# << turn on when active
      @class Oracle
      @brief Direct interface to Oracle database.
      @param connect String containing connection parameters.

          The connect string uses the standard connection values:
          - username
          - password
          - database

          Oracle does not use ports in its connection string. That information
          is generally stored in your TNSNAMES.ora file.
          */

        FALCON_FUNC Oracle_init( VMachine *vm )
        {
            Item *paramsI = vm->param(0);
            Item *i_tropts = vm->param(1);
            if ( ! paramsI || ! paramsI->isString() || ( i_tropts && ! i_tropts->isString() ) )
            {
                throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                        .extra( "S,[S]" ) );
            }

            String *params = paramsI->asString();

            DBIHandle *hand = 0;
            try
            {
                //hand = theOracleService.connect( *params );
                if( i_tropts != 0 )
                {
                    //  hand->options( *i_tropts->asString() );
                }

                //CoreObject *instance = theOracleService.makeInstance( vm, hand );
                //vm->retval( instance );
            }
            catch (...)
            {
                //delete hand;
                throw ;
            }
        }

    } /* namespace Ext */
} /* namespace Falcon */

/* end of oracle_ext.cpp */


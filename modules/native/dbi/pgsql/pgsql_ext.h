/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_ext.h
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

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef PGSQL_EXT_H
#define PGSQL_EXT_H

namespace Falcon {

class VMachine;

namespace Ext {

FALCON_FUNC PgSQL_init( VMachine* vm );
FALCON_FUNC PgSQL_prepareNamed( VMachine* vm );

} // !Ext
} // !Falcon

#endif /* PGSQL_EXT_H */

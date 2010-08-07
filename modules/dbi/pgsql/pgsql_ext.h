/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_ext.h
 *
 * PgSQL Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Sun Dec 23 21:48:48 2007
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

namespace Falcon
{

class VMachine;

namespace Ext
{

FALCON_FUNC PgSQL_init( VMachine *vm );

}
}

#endif /* PGSQL_EXT_H */

/* end of pgsql_ext.h */


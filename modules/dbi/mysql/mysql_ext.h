/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_ext.h
 *
 * MySQL Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 21:35:18 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

/*
#include <falcon/setup.h>
#include <falcon/types.h>
*/
#include <falcon/module.h>

#ifndef MYSQL_EXT_H
#define MYSQL_EXT_H

namespace Falcon
{

class VMachine;

namespace Ext
{

FALCON_FUNC MySQL_init( VMachine *vm );

}
}

#endif /* MYSQL_EXT_H */

/* end of mysql_ext.h */


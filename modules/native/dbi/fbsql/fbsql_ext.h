/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_ext.h
 *
 * Firebird database server Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 20 Sep 2010 21:08:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

/*
#include <falcon/setup.h>
#include <falcon/types.h>
*/
#include <falcon/module.h>

#ifndef FALCON_FIREBIRD_EXT_H
#define FALCON_FIREBIRD_EXT_H

namespace Falcon
{

class VMachine;

namespace Ext
{

FALCON_FUNC Firebird_init( VMachine *vm );

}
}

#endif /* FALCON_FIREBIRD_EXT_H */

/* end of fbsql_ext.h */


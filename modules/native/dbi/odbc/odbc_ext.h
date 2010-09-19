/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_ext.h
 *
 * MySQL Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Tiziano De Rubeis
 * Begin: Tue Sep 30 17:00:00 2008
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

#ifndef ODBC_EXT_H
#define ODBC_EXT_H

namespace Falcon
{

class VMachine;

namespace Ext
{

FALCON_FUNC ODBC_init( VMachine *vm );

}
}

#endif /* ODBC_EXT_H */



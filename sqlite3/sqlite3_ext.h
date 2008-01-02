/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.h
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Wed Jan 02 16:47:15 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef SQLITE3_EXT_H
#define SQLITE3_EXT_H

namespace Falcon
{

class VMachine;

namespace Ext
{

FALCON_FUNC SQLite3_init( VMachine *vm );

}
}

#endif /* SQLITE3_EXT_H */

/* end of sqlite3_ext.h */


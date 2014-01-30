/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_ext.cpp
 *
 * Firebird Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Mon, 20 Sep 2010 21:02:16 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/engine.h>

#include "fbsql_mod.h"
#include "fbsql_ext.h"

/*#
   @beginmodule dbi.fbsql
*/
namespace Falcon {
namespace Ext {

/*#
   @class Firebird
   @brief Direct interface to Firebird database.
   @param connect String containing connection parameters.

   The connect string uses the standard connection values:
   - uid: user id
   - pwd: password
   - db: database where to connect
   - host: host where to connect (defaults to localhost)
   - port: prot where to connect (defaults to mysql standard port)

   Following connection options are specific of the Firebird database engine,
   and require a value to be passed in the form of
   @code
      "fbsql:...;key=value;key=value"
   @endcode

   - epwd: Encrypted password
   - role: Role
   - sa: System Administrator user name
   - license: Authorization key for a software license
   - ekey: Database encryption key
   - nbuf: Number of cache buffers
   - kscope: dbkey context scope
   - lcmsg: Language-specific message file
   - lctype: Character set to be utilized
   - tout: Connection timeout


   Following connection options are specific of the Firebird database engine,
   and require a boolean value that can be either "yes" or "no".

   - reserve: Specify whether or not to reserve a small amount of space on each database
              page for holding backup versions of records when modifications are made
   - dmg: Specify whether or not the database should be marked as damaged
   - verify: Perform consistency checking of internal structures
   - shadow: Activate the database shadow, an optional, duplicate, in-sync copy of the database
   - delshadow: Delete the database shadow
   - beginlog: Activate a replay logging system to keep track of all database calls
   - quitlog: Deactivate the replay logging system

@section fb_opt Transaction options.
   Other than the default options provided by @a Handle.options, the Firebird driver
   provides the following transaction options:

   - getaffected: Reads affected rows after every query/execute operation. Can be 'on' or 'off'
     (defaults on).
*/

ClassFBSQLDBIHandle::ClassFBSQLDBIHandle():
         ClassDriverDBIHandle("FBSql")
{
}

ClassFBSQLDBIHandle::~ClassFBSQLDBIHandle()
{
}

void* ClassFBSQLDBIHandle::createInstance() const
{
   return new DBIHandleFB(this);
}


} /* namespace Ext */
} /* namespace Falcon */

/* end of fbsql_ext.cpp */


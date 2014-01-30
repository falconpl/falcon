/*
 * FALCON - The Falcon Programming Language.
 * FILE: sqlite3_ext.h
 *
 * SQLite3 Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Thu, 30 Jan 2014 13:47:51 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_DBI_SQLITE3_EXT_H
#define FALCON_DBI_SQLITE3_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/dbi_driverclass.h>

namespace Falcon
{
namespace Ext
{

class ClassSqlite3DBIHandle: public ClassDriverDBIHandle
{
public:
   ClassSqlite3DBIHandle();
   virtual ~ClassSqlite3DBIHandle();
   virtual void* createInstance() const;
};

}
}

#endif /* SQLITE3_EXT_H */

/* end of sqlite3_ext.h */


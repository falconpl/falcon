/*
 * FALCON - The Falcon Programming Language.
 * FILE: pgsql_ext.h
 *
 * PgSQL Falcon service/driver
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar, Stanislas Marquis, Giancarlo Niccolai
 * Begin: Sun Dec 23 21:54:42 2007
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#include <falcon/setup.h>
#include <falcon/types.h>

#ifndef FALCON_DBI_PGSQL_EXT_H
#define FALCON_DBI_PGSQL_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/dbi_driverclass.h>

namespace Falcon {
namespace Ext {

class ClassPGSQLDBIHandle: public ClassDriverDBIHandle
{
public:
   ClassPGSQLDBIHandle();
   virtual ~ClassPGSQLDBIHandle();
   virtual void* createInstance() const;
};

}
}

#endif /* PGSQL_EXT_H */

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

#ifndef FALCON_DBI_MYSQL_EXT_H
#define FALCON_DBI_MYSQL_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/dbi_driverclass.h>

namespace Falcon {
namespace Ext {

class ClassMySQLDBIHandle: public ClassDriverDBIHandle
{
public:
   ClassMySQLDBIHandle();
   virtual ~ClassMySQLDBIHandle();
   virtual void* createInstance() const;
};

}
}

#endif /* FALCON_DBI_MYSQL_EXT_H */

/* end of mysql_ext.h */

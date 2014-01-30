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

#ifndef FALCON_FIREBIRD_EXT_H
#define FALCON_FIREBIRD_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/dbi_driverclass.h>

namespace Falcon {
namespace Ext {

class ClassFBSQLDBIHandle: public ClassDriverDBIHandle
{
public:
   ClassFBSQLDBIHandle();
   virtual ~ClassFBSQLDBIHandle();
   virtual void* createInstance() const;
};

}
}

#endif /* FALCON_FIREBIRD_EXT_H */

/* end of fbsql_ext.h */

/*
 * FALCON - The Falcon Programming Language.
 * FILE: odbc_ext.h
 *
 * ODBC Falcon extension interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Tue Sep 30 17:00:00 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_DBI_ODBC_EXT_H
#define FALCON_DBI_ODBC_EXT_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/dbi_driverclass.h>

namespace Falcon {
namespace Ext {

class ClassODBCDBIHandle: public ClassDriverDBIHandle
{
public:
   ClassODBCDBIHandle();
   virtual ~ClassODBCDBIHandle();
   virtual void* createInstance() const;
};

}
}

#endif /* ODBC_EXT_H */



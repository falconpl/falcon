/*
   FALCON - The Falcon Programming Language.
   FILE: pgsql_fm.h

   Postgre driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DBI_PGSQL_H_
#define _FALCON_DBI_PGSQL_H_

#include <falcon/dbi_drivermod.h>

namespace Falcon {

class PGSQLDBIModule: public DriverDBIModule
{
public:
   PGSQLDBIModule();
   virtual ~PGSQLDBIModule();
};

}

#endif

/* end of sqlite3_fm.h */

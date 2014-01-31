/*
   FALCON - The Falcon Programming Language.
   FILE: odbc_fm.h

   ODBC driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DBI_ODBC_FM_H_
#define _FALCON_DBI_ODBC_FM_H_

#include <falcon/dbi_drivermod.h>

namespace Falcon {

class ODBCDBIModule: public DriverDBIModule
{
public:
   ODBCDBIModule();
   virtual ~ODBCDBIModule();
};

}

#endif

/* end of odbc_fm.h */

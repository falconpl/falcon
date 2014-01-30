/*
   FALCON - The Falcon Programming Language.
   FILE: sqlite3_fm.cpp

   SQLite3 driver main module

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 May 2010 18:25:58 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DBI_SQLITE3_H_
#define _FALCON_DBI_SQLITE3_H_

#include <falcon/dbi_drivermod.h>

namespace Falcon {

class Sqlite3DBIModule: public DriverDBIModule
{
public:
   Sqlite3DBIModule();
   virtual ~Sqlite3DBIModule();
};

}

#endif

/* end of sqlite3_fm.h */

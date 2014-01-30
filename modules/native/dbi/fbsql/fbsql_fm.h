/*
 * FALCON - The Falcon Programming Language.
 * FILE: fbsql_fm.h
 *
 * Firebird SQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Thu, 30 Jan 2014 14:58:04 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_DBI_FBSQL_FM_H_
#define _FALCON_DBI_FBSQL_FM_H_

#include <falcon/module.h>
#include <falcon/dbi_drivermod.h>

namespace Falcon
{

class ModuleFBSQL: public DriverDBIModule
{
public:
   ModuleFBSQL();
   virtual ~ModuleFBSQL();
};

}

#endif

/* end of fbsql_fm.h */

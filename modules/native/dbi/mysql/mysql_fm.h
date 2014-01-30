/*
 * FALCON - The Falcon Programming Language.
 * FILE: mysql_fm.h
 *
 * MySQL driver main module interface
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sun, 23 May 2010 16:58:53 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef _FALCON_DBI_MYSQL_FM_H_
#define _FALCON_DBI_MYSQL_FM_H_

#include <falcon/module.h>
#include <falcon/dbi_drivermod.h>

namespace Falcon
{

class ModuleMySQL: public DriverDBIModule
{
public:
   ModuleMySQL();
   virtual ~ModuleMySQL();
};

}

#endif

/* end of mysql_fm.h */

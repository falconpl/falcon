/*
   FALCON - The Falcon Programming Language
   FILE: dbi_mod.h

   DBI module -- module service classes
   -------------------------------------------------------------------
   Author: Jeremy Cowgar
   Begin: 2007-12-22 10:06
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   dbi_mod.h - DBI module -- module service classes
*/

#ifndef flc_dbi_mod_H
#define flc_dbi_mod_H

#include <falcon/string.h>

namespace Falcon {

enum
{
   SQL_OK = 0,
};

class DBIConnection;
class DBIResult;

#endif

/* end of dbi_mod.h */

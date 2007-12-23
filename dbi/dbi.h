/*
   FALCON - The Falcon Programming Language.
   FILE: dbi.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 20:33:57 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef DBI_H
#define DBI_H

#include "../include/dbiservice.h"
#include <falcon/flcloader.h>

namespace Falcon
{

class DBILoaderImpl: public DBILoader
{
   FlcLoader m_loader;

public:
   DBILoaderImpl();
   ~DBILoaderImpl();

   virtual DBIService *loadDbProvider( VMachine *vm, const String &provName );

};

// Singleton instance.
extern Falcon::DBILoaderImpl theDBIService;

}

#endif

/* end of dbi.h */


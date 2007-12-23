/*
   FALCON - The Falcon Programming Language.
   FILE: mysql_ext.cpp

   MySQL Falcon extension interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 23 Dec 2007 22:02:37 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/engine.h>
#include "mysql.h"
#include "mysql_ext.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC MySQL_init( VMachine *vm )
{
   // initialize here the instance.
   // we have been already provided with a MySQL empty object,
   // which is already derived from DBIHandler.
   CoreObject *self = vm->self().asObject();

   // So, all we have to do is...
   DBIService::dbi_status status;
   DBIHandleMysql *myhdb =
      static_cast<DBIHandleMysql *>( theMySQLService.connect( "Some connection data", false, status ) );

   if ( myhdb == 0 )
   {
      // raise error.
      // Btw, It would be cute to raise a DBIError declared in dbi.
      return;
   }

   // do eventually some initialization on myhdb.
   self->setUserData( myhdb );
}

// if we don't have special methods for the Falcon MySQL class, all is already done
// in the base class.

}
}

/* end of mysql_ext.cpp */


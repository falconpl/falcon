/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

   Falcon VM interface to logging module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to configuration parser module.
*/


#include <falcon/fassert.h>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/carray.h>
#include <falcon/lineardict.h>
#include <falcon/stream.h>
#include <falcon/memory.h>

#include "logging_ext.h"
#include "logging_mod.h"
#include "logging_st.h"

/*#
   @beginmodule feather_configparser
*/

namespace Falcon {
namespace Ext {

// ==============================================
// Class LogArea
// ==============================================

/*#
   @class LogArea
   @brief Collection of log channels.
   @param name The name of the area.
*/


FALCON_FUNC  LogArea_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Item *i_aname = vm->param(0);

   if ( i_aname == 0 || ! i_aname->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   self->setUserData( new LogArea( *i_aname->asString() ) );
}

}
}

/* end of logging_ext.cpp */

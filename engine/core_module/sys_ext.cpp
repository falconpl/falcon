/*
   FALCON - The Falcon Programming Language.
   FILE: sys_ext.cpp

   System relevant extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/vmevent.h>

namespace Falcon {
namespace core {

/*#
   @function exit
   @ingroup core_syssupport
   @param value an item representing VM exit code.
   @brief Requires immediate termination of the program.

   The VM execution will be interrupted, with normal state, and the item will be
   passed in the A register of the VM, where embedding application expects to receive
   the exit value. Semantic of the exit value will vary depending on the embedding
   application. The Falcon command line tools (and so, stand alone scripts) will pass
   this return value to the system if it is an integer, else they will terminate
   passing 0 to the system.

   This function terminates the VM also when there are more coroutines running.

*/

FALCON_FUNC  core_exit ( ::Falcon::VMachine *vm )
{
   Item *ret = vm->param(0);

   if ( ret != 0 )
      vm->retval( *ret );
   else
      vm->retnil();
   
   throw VMEventQuit();
}

}
}

/* end of sys_ext.cpp */

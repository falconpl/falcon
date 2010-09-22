/*
   FALCON - The Falcon Programming Language.
   FILE: conbox_ext.cpp

   Basic Console I/O support
   Interface extension functions
   -------------------------------------------------------------------
   Author: Unknown author
   Begin: Thu, 05 Sep 2008 20:12:14 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: The above AUTHOR

      Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

      http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

/** \file
   Basic Console I/O support
   Interface extension functions
*/

#include <falcon/engine.h>
#include "conbox_ext.h"
#include "conbox_st.h"

namespace Falcon {
namespace Ext {

// The following is a faldoc block for the function
/*#
   @function skeleton
   @brief A basic script function.
   @return Zero.


*/

FALCON_FUNC  cb_skeleton( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) 0 );
}

/*#
   @function skeletonString
   @brief A function returning a string in the string table.
   @return A message that can be internationalized.

   This function returns a string from the string table
   (see conbox_st.h).

   The returned string may be internationalized through
   the standard falcon internationalization system (the
   same available for scripts).

   A real module will want to use this system to produce
   locale-configurable messages (expecially the "extra"
   field of error descriptions).
*/
FALCON_FUNC  cb_skeletonString( ::Falcon::VMachine *vm )
{
   vm->retval( FAL_STR( conbox_msg_1 ) );
}

}
}

/* end of conbox_ext.cpp */

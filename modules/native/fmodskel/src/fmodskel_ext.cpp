/*
   @{fmodskel_MAIN_PRJ}@
   FILE: @{fmodskel_PROJECT_NAME}@_ext.cpp

   @{fmodskel_DESCRIPTION}@
   Interface extension functions
   -------------------------------------------------------------------
   Author: @{fmodskel_AUTHOR}@
   Begin: @{fmodskel_DATE}@

   -------------------------------------------------------------------
   (C) Copyright @{fmodskel_YEAR}@: @{fmodskel_COPYRIGHT}@

   @{fmodskel_LICENSE}@
*/

/** \file
   @{fmodskel_DESCRIPTION}@
   Interface extension functions
*/

#include <falcon/engine.h>
#include "@{fmodskel_PROJECT_NAME}@_mod.h"
#include "@{fmodskel_PROJECT_NAME}@_ext.h"
#include "@{fmodskel_PROJECT_NAME}@_st.h"

namespace Falcon { namespace Ext {

// The following is a faldoc block for the function
/*#
   @function skeleton
   @brief A basic script function.
   @return Zero.

   This function just illustrates how to bind the ineer MOD logic
   with the script. Also, Mod::skeleton(), used by this
   function, is exported through the "service", so it is
   possible to call the MOD logic directly from an embedding
   application, once loaded the module and accessed the service.
*/

FALCON_FUNC  skeleton( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) Mod::skeleton() );
}

/*#
   @function skeletonString
   @brief A function returning a string in the string table.
   @return A message that can be internationalized.

   This function returns a string from the string table
   (see @{fmodskel_PROJECT_NAME}@_st.h).

   The returned string may be internationalized through
   the standard falcon internationalization system (the
   same available for scripts).

   A real module will want to use this system to produce
   locale-configurable messages (expecially the "extra"
   field of error descriptions).
*/
FALCON_FUNC  skeletonString( ::Falcon::VMachine *vm )
{
   vm->retval( FAL_STR( @{fmodskel_PROJECT_NAME}@_msg_1 ) );
}

}} // namespace Falcon::Ext

/* end of @{fmodskel_PROJECT_NAME}@_ext.cpp */

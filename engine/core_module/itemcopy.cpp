/*
   FALCON - The Falcon Programming Language.
   FILE: itemcopy.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio mar 16 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   ItemCopy API function
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/item.h>
#include <falcon/carray.h>
#include <falcon/cdict.h>
#include <falcon/cobject.h>
#include <falcon/vm.h>

/*#
   @beginmodule falcon_rtl
*/

namespace Falcon {

namespace Ext {

/*#
   @function itemCopy
   @inset rtl_general_purpose
   @brief Performs a shallow copy of one item.
   @param item The item to be copied.
   @return A copy of the item.
   @raise CloneError if the item is not cloneable.

   This function works as the FBOM method @a BOM.clone.
   @note Deprecated; used the BOM method instead.
*/
FALCON_FUNC  itemCopy( ::Falcon::VMachine *vm )
{
   Item *item = vm->param(0);
   if ( item == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ) ) );
      return;
   }

   if ( ! item->clone( vm->regA(), vm ) )
   {
      vm->raiseError( new CloneError( ErrorParam( e_uncloneable, __LINE__ ).origin( e_orig_runtime ) ) );
   }
}

}

}

/* end of itemcopy.cpp */

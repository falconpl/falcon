/*
   FALCON - The Falcon Programming Language.
   FILE: itemcopy.cpp
   $Id: itemcopy.cpp,v 1.6 2007/08/03 13:17:07 jonnymind Exp $

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio mar 16 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

namespace Falcon {

namespace Ext {

/**
   itemCopy( item ) --> shallow copy
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

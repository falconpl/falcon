/*
   FALCON - The Falcon Programming Language.
   FILE: membuf_ext.cpp

   Memory buffer functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"

/*#
   @beginmodule core
*/
namespace Falcon {
namespace core {

FALCON_FUNC Make_MemBuf( ::Falcon::VMachine *vm )
{
   Item *i_size = vm->param(0);
   Item *i_wordSize = vm->param(1);

   if( ( i_size == 0 || ! i_size->isOrdinal() ) ||
       ( i_wordSize != 0 && ! i_wordSize->isOrdinal() )
      )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N,[N]" ) ) );
      return;
   }

   int64 wordSize = i_wordSize == 0 ? 1: i_wordSize->forceInteger();
   int64 size = i_size->forceInteger();
   if ( wordSize < 1 || wordSize > 4 || size <= 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ) ) );
      return;
   }

   MemBuf *mb = 0;
   switch( wordSize )
   {
      case 1: mb = new MemBuf_1( vm, (uint32) size ); break;
      case 2: mb = new MemBuf_2( vm, (uint32) size * 2); break;
      case 3: mb = new MemBuf_3( vm, (uint32) size * 3); break;
      case 4: mb = new MemBuf_4( vm, (uint32) size * 4); break;
   }
   fassert( mb != 0 );
   vm->retval( mb );
}

}
}

/* end of membuf_ext.cpp */

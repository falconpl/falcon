/*
   FALCON - The Falcon Programming Language.
   FILE: poopcomp_ext.cpp

   Prototype oop oriented sequence interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 10 Aug 2009 11:19:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/coreobject.h>
#include <falcon/coredict.h>
#include <falcon/poopseq.h>
#include "core_module.h"

namespace Falcon {
namespace core {

FALCON_FUNC  Object_comp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "R|A|C|Sequence, [C]" ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreObject* obj = vm->self().asObject();

   Item i_gen = *vm->param(0);
   Item i_check = vm->param(1) == 0 ? Item(): *vm->param(1);
   PoopSeq seq( vm, Item(obj) );  // may throw
   seq.comprehension( vm, i_gen, i_check );
   vm->retval( vm->self() );
}

}
}

/* end of poopseq_ext.cpp */

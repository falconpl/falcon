/*
   FALCON - The Falcon Programming Language.
   FILE: continuation.h

   Continuation object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Dec 2009 17:04:27 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/continuation.h>
#include <falcon/mempool.h>
#include <falcon/globals.h>

namespace Falcon {
namespace core {


//===========================================================
//

FALCON_FUNC Continuation_init ( ::Falcon::VMachine *vm )
{
   Item* i_cc = vm->param(0);
   if ( i_cc == 0 || ! i_cc->isCallable() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).
            extra("C") );
   }

   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   cc->cont( new Continuation(vm) );
   cc->ccItem( *i_cc );
}


FALCON_FUNC Continuation_call ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   // call in passive phase calls the desired item.
   if ( ! cc->cont()->phase() )
   {
      cc->cont()->callMark();
      if( cc->cont()->jump() )
         return;

      // otherwise, we have to call the item from here.
      vm->pushParam( cc );
      vm->callItem( cc->ccItem(), 1 );
   }
   else
   {
      // in passive phase, we must return
      cc->cont()->invoke( vm->param(0) == 0 ? Item(): *vm->param(0) );
   }
}

FALCON_FUNC Continuation_reset ( ::Falcon::VMachine *vm )
{
   ContinuationCarrier* cc = dyncast<ContinuationCarrier*>( vm->self().asObject() );
   cc->cont()->reset();
}


}
}

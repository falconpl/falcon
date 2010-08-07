/*
   FALCON - The Falcon Programming Language.
   FILE: funcext_ext.h

   Functional extensions - extension definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler module main file - extension definitions.
*/

#ifndef flc_compiler_ext_H
#define flc_compiler_ext_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error_base.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC  fe_at ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_ge ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_gt ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_le ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_lt ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_eq ( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_neq( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_deq( ::Falcon::VMachine *vm );

FALCON_FUNC  fe_add( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_sub( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_mul( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_div( ::Falcon::VMachine *vm );
FALCON_FUNC  fe_mod( ::Falcon::VMachine *vm );

}
}

#endif

/* end of funcext_ext.h */

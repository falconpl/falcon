/*
   FALCON - The Falcon Programming Language.
   FILE: compiler_ext.h

   Compiler module main file - extension definitions.
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

namespace Falcon {

namespace Ext {

FALCON_FUNC Compiler_init( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_compile( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_loadByName( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_loadModule( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_setDirective( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_addFalconPath( ::Falcon::VMachine *vm );


FALCON_FUNC Module_get( ::Falcon::VMachine *vm );
FALCON_FUNC Module_set( ::Falcon::VMachine *vm );
FALCON_FUNC Module_getReference( ::Falcon::VMachine *vm );
FALCON_FUNC Module_unload( ::Falcon::VMachine *vm );
FALCON_FUNC Module_engineVersion( ::Falcon::VMachine *vm );
FALCON_FUNC Module_moduleVersion( ::Falcon::VMachine *vm );

}
}

#endif

/* end of compiler_ext.h */

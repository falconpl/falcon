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
#include <falcon/error_base.h>

#ifndef FALCON_COMPILER_ERROR_BASE
   #define FALCON_COMPILER_ERROR_BASE        1000
#endif

#define FALCOMP_ERR_UNLOADED   (FALCON_COMPILER_ERROR_BASE + 0)

namespace Falcon {

namespace Ext {

FALCON_FUNC BaseCompiler_setDirective( ::Falcon::VMachine *vm );
FALCON_FUNC BaseCompiler_addFalconPath( ::Falcon::VMachine *vm );

FALCON_FUNC Compiler_init( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_compile( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_loadByName( ::Falcon::VMachine *vm );
FALCON_FUNC Compiler_loadFile( ::Falcon::VMachine *vm );

FALCON_FUNC ICompiler_init( ::Falcon::VMachine *vm );
FALCON_FUNC ICompiler_compileNext( ::Falcon::VMachine *vm );
FALCON_FUNC ICompiler_compileAll( ::Falcon::VMachine *vm );
FALCON_FUNC ICompiler_reset( ::Falcon::VMachine *vm );



FALCON_FUNC Module_globals( ::Falcon::VMachine *vm );
FALCON_FUNC Module_exported( ::Falcon::VMachine *vm );
FALCON_FUNC Module_get( ::Falcon::VMachine *vm );
FALCON_FUNC Module_set( ::Falcon::VMachine *vm );
FALCON_FUNC Module_getReference( ::Falcon::VMachine *vm );
FALCON_FUNC Module_unload( ::Falcon::VMachine *vm );
FALCON_FUNC Module_engineVersion( ::Falcon::VMachine *vm );
FALCON_FUNC Module_moduleVersion( ::Falcon::VMachine *vm );
FALCON_FUNC Module_attributes( ::Falcon::VMachine *vm );

}
}

#endif

/* end of compiler_ext.h */

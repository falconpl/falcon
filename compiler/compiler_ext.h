/*
   FALCON - The Falcon Programming Language.
   FILE: compiler_ext.h
   $Id: compiler_ext.h,v 1.3 2007/07/25 19:10:39 jonnymind Exp $

   Compiler module main file - extension definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007
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

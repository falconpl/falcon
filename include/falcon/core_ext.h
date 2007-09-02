/*
   FALCON - The Falcon Programming Language.
   FILE: core_func.h
   $Id: core_ext.h,v 1.2 2007/03/04 17:39:02 jonnymind Exp $

   Falcon module manager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef flc_COREFUNC_H
#define flc_COREFUNC_H

namespace Falcon {

class VMachine;

/** Core module namespace
   This namespace contains the extension functions that the falcon scripts
   are expecting to be able to access in every circumstance. They are
   mainly harmless functions that allow item manipulations, and they
   can usually be considere "language standards". Anyhow, it is possible
   that the embedding application may wish to provide a limited set of this,
   or a personalized version. In this case, it will provide a different
   core moudle by using something similar to the core_module_init() function.
*/

namespace core {

FALCON_FUNC len ( ::Falcon::VMachine *vm );
FALCON_FUNC DgetKeyAt ( ::Falcon::VMachine *vm );
FALCON_FUNC DgetValueAt ( ::Falcon::VMachine *vm );
FALCON_FUNC DgetPairAt( ::Falcon::VMachine *vm );

/** Useful symbol to be exported by the engine DLL */
FALCON_FUNC_DYN_SYM CreateTraceback( ::Falcon::VMachine *vm );
/** Useful symbol to be exported by the engine DLL */
FALCON_FUNC_DYN_SYM TraceStep( ::Falcon::VMachine *vm );
/** Useful symbol to be exported by the engine DLL */
FALCON_FUNC_DYN_SYM TraceStep_toString( ::Falcon::VMachine *vm );
/** Useful symbol to be exported by the engine DLL */
FALCON_FUNC_DYN_SYM Error( ::Falcon::VMachine *vm );
/** Useful symbol to be exported by the engine DLL */
FALCON_FUNC_DYN_SYM Error_toString( ::Falcon::VMachine *vm );
/** Useful symbol to be exported by the engine DLL
     This is exported also by error.h
 */
FALCON_FUNC_DYN_SYM Error_init( ::Falcon::VMachine *vm );
} // end of core namespace

FALCON_DYN_SYM Module * core_module_init();

extern Module *core_module;

}

#endif

/* end of core_func.h */

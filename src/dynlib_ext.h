/*
   The Falcon Programming Language
   FILE: dynlib_ext.cpp

   Direct dynamic library interface for Falcon
   Interface extension functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 28 Oct 2008 22:23:29 +0100

   -------------------------------------------------------------------
   (C) Copyright 2008: The Falcon Comittee

   See the LICENSE file distributed with this package for licensing details.
*/

/** \file
   Direct dynamic library interface for Falcon
   Interface extension functions - header file
*/

#ifndef dynlib_ext_H
#define dynlib_ext_H

#include <falcon/module.h>

#ifndef FALCON_DYNLIB_ERROR_BASE
#define FALCON_DYNLIB_ERROR_BASE 2250
#endif

namespace Falcon {
namespace Ext {

FALCON_FUNC  limitMembuf( ::Falcon::VMachine *vm );
FALCON_FUNC  limitMembufW( ::Falcon::VMachine *vm );
FALCON_FUNC  derefPtr( ::Falcon::VMachine *vm );
FALCON_FUNC  dynExt( ::Falcon::VMachine *vm );

FALCON_FUNC  stringToPtr( ::Falcon::VMachine *vm );
FALCON_FUNC  memBufToPtr( ::Falcon::VMachine *vm );
FALCON_FUNC  memBufFromPtr( ::Falcon::VMachine *vm );
FALCON_FUNC  getStruct( ::Falcon::VMachine *vm );
FALCON_FUNC  setStruct( ::Falcon::VMachine *vm );
FALCON_FUNC  memSet( ::Falcon::VMachine *vm );

FALCON_FUNC  DynLib_init( ::Falcon::VMachine *vm );
FALCON_FUNC  DynLib_get( ::Falcon::VMachine *vm );
FALCON_FUNC  DynLib_query( ::Falcon::VMachine *vm );
FALCON_FUNC  DynLib_unload( ::Falcon::VMachine *vm );


FALCON_FUNC  Dyn_dummy_init( ::Falcon::VMachine *vm );
FALCON_FUNC  DynFunction_call( ::Falcon::VMachine *vm );
FALCON_FUNC  DynFunction_toString( ::Falcon::VMachine *vm );

FALCON_FUNC  DynOpaque_toString( ::Falcon::VMachine *vm );
FALCON_FUNC  DynOpaque_getData( ::Falcon::VMachine *vm );

FALCON_FUNC  testParser( ::Falcon::VMachine *vm );

//=====================
// DynLib Error class
//=====================

FALCON_FUNC DynLibError_init( VMachine *vm );

}
}

#endif

/* end of dynlib_ext.h */

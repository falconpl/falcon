/*
 * FALCON - The Falcon Programming Language
 * FILE: zlib_ext.cpp
 *
 * zlib module main file - extension definitions
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Thu Jan 3 2007
 * -------------------------------------------------------------------
 * (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in it's compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes bundled with this
 * package.
 */

#include <falcon/engine.h>

#include "zlib.h"
#include "zlib_ext.h"

namespace Falcon {
namespace Ext {

FALCON_FUNC ZLib_init( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   GarbageString *gsVersion = new GarbageString( vm, zlibVersion() );
   self->setProperty( "version", gsVersion );

   vm->retval( self );
}

}
}


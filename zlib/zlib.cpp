/*
 * FALCON - The Falcon Programming Language
 * FILE: zlib.cpp
 *
 * zlib module main file
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

/**
 * \file
 * This module exports zlib and module loader facility to falcon
 * scripts.
 */

#include <falcon/module.h>
#include "zlib_ext.h"

#include "version.h"

FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "zlib" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   Falcon::Symbol *c_zlib = self->addClass( "ZLib", Falcon::Ext::ZLib_init );
   self->addClassMethod( c_zlib, "compress", Falcon::Ext::ZLib_compress );
   self->addClassMethod( c_zlib, "uncompress", Falcon::Ext::ZLib_uncompress );
   self->addClassProperty( c_zlib, "version" );

   return self;
}

/* end of zlib.cpp */


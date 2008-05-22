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
/*#
   @module feather_zlib The ZLib module
   @brief Minimal compress/uncompress functions.
   
   This module provides an essential interface to the Zlib compression routines.
   
   The greatest part of the functionalites of the module are encapsulated in the
   @a ZLib class, which provided some class-wide methods to compress and uncompress
   data.

   The following example can be considered a minimal usage pattern:
   @code
   load zlib

   original = "Mary had a little lamb, it's fleece was white as snow."
   > "Uncompressed: ", original

   comped = ZLib.compress( original )
   > "Compressed then uncompressed:"
   > "   ", ZLib.uncompress( comped )

   @endcode

   The module declares also a @b ZLibError that is raised in case of
   failure in compression/decompression operations.
*/


FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "zlib" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   Falcon::Symbol *c_zlib = self->addClass( "ZLib" );
   self->addClassMethod( c_zlib, "compress", Falcon::Ext::ZLib_compress );
   self->addClassMethod( c_zlib, "uncompress", Falcon::Ext::ZLib_uncompress );
   self->addClassMethod( c_zlib, "getVersion", Falcon::Ext::ZLib_getVersion );

   //============================================================
   // ZlibError class
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *procerr_cls = self->addClass( "ZLibError", Falcon::Ext::ZlibError_init );
   procerr_cls->setWKS( true );
   procerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   return self;
}

/* end of zlib.cpp */


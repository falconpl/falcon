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
#include "zlib.h"

#include "version.h"
/*#
   @module zlib ZLib basic binding.
   @ingroup feathers
   @brief Compress/uncompress functions (zlib binding).

   The @b ZLib module provides an essential interface to the Zlib compression routines.

   The greatest part of the functionalites of the module are encapsulated in the
   @a ZLib class, which provided some class-wide methods to compress and uncompress
   data.

   The following example can be considered a minimal usage pattern:
   @code
   load zlib

   original = "Mary had a little lamb, it's fleece was white as snow."
   > "Uncompressed: ", original

   comped = ZLib.compressText( original )
   > "Compressed then uncompressed:"
   > "   ", ZLib.uncompressText( comped )

   @endcode

   The module declares also a @b ZLibError that is raised in case of
   failure in compression/decompression operations.

   @beginmodule zlib
*/

FALCON_MODULE_DECL
{
   Falcon::Module* mod = new ::Falcon::Feathrs::ModuleZLib;
   return mod;
}

/* end of zlib.cpp */



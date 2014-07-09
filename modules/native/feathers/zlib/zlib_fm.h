/*
 * FALCON - The Falcon Programming Language
 * FILE: zlib_fm.h
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

#ifndef FALCON_FEATHERS_ZLIB_FM_H
#define FALCON_FEATHERS_ZLIB_FM_H

#define FALCON_FEATHER_ZLIB_NAME "zlib"

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleZLib: public Falcon::Module
{
public:
   ModuleZLib();
   virtual ~ModuleZLib();
};

}}

#endif

/* end of zlib_fm.h */

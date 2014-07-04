/*
 * FALCON - The Falcon Programming Language
 * FILE: zlib_fm.cpp
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


#include "zlib_ext.h"
#include "zlib_fm.h"
#include "version.h"

namespace Falcon {
namespace Feathers {

ModuleZLib::ModuleZLib():
   Module("zlib")
{

   //====================================
   // ZLib class

   addMantra( new Falcon::Ext::Function_getVersion);
   addMantra( new Falcon::Ext::Function_compress);
   addMantra( new Falcon::Ext::Function_compressText);
   addMantra( new Falcon::Ext::Function_uncompress);
   addMantra( new Falcon::Ext::Function_uncompressText);


   ////============================================================
   //// ZlibError class

   addMantra(Falcon::Ext::ClassZLibError::singleton());
}

ZLibModule::~ZLibModule() {};

}}

/* end of zlib_fm.cpp */



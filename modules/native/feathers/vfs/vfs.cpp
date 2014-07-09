/*
   FALCON - The Falcon Programming Language.
   FILE: vfs.cpp

   Interface to Falcon Virtual File System -- main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "vfs_fm.h"

//=================================
// Export function.

/*# @module vfs Virtual File System
 @brief Interface for abstract access to local and remote filesystems.
 @ingroup feathers
 */

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL 
{
   Falcon::Module* mod = new Falcon::Feathers::ModuleVFS;
   return mod;
}

#endif

/* end of vfs.cpp */

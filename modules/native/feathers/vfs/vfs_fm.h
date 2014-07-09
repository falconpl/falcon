/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_fm.h

   Interface to Falcon Virtual File System -- main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_FEATHERS_VFS_FM_H
#define FALCON_FEATHERS_VFS_FM_H

#define FALCON_FEATHER_VFS_NAME "vfs"

#include <falcon/module.h>

namespace Falcon {
namespace Feathers {

class ModuleVFS: public Module
{
public:
   ModuleVFS();
   virtual ~ModuleVFS();
};

}}

#endif

/* end of vfs_fm.h */

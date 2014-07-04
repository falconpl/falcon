/*
   FALCON - The Falcon Programming Language.
   FILE: vfs_fm.cpp

   Interface to Falcon Virtual File System -- main file.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 24 Oct 2011 14:34:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/moduels/native/feathers/vfs_fm.cpp"

#include <falcon/vfsprovider.h>

#include "vfs_fm.h"
#include "vfs_ext.h"
#include "classvfs.h"
#include "classfilestat.h"
#include "classdirectory.h"

namespace Falcon {
namespace Feathers {

ModuleVFS::ModuleVFS():
   Module("vfs")
{
   *this
      // Standard functions
      << new Ext::ClassVFS
      << new Ext::ClassFileStat
      << new Ext::ClassDirectory
      << new Ext::Function_IOStream
      << new Ext::Function_InputStream
      << new Ext::Function_OutputStream
      
      // Standard classes
      ;
   
   // constants
   this->addConstant( "O_RD", (int64)VFSProvider::OParams::e_oflag_rd );
   this->addConstant( "O_WR", (int64)VFSProvider::OParams::e_oflag_wr );
   this->addConstant( "O_APPEND", (int64)VFSProvider::OParams::e_oflag_append );
   this->addConstant( "O_TRUNC", (int64)VFSProvider::OParams::e_oflag_trunc );
   //this->addConstant( "O_RAW", (int64)FALCON_VFS_MODE_FLAG_RAW );
   
   this->addConstant( "SH_NR", (int64)VFSProvider::OParams::e_sflag_nr );
   this->addConstant( "SH_NW", (int64)VFSProvider::OParams::e_sflag_nw );

   this->addConstant( "C_NOOVR", (int64)VFSProvider::CParams::e_cflag_noovr );
   this->addConstant( "C_NOSTREAM", (int64)VFSProvider::CParams::e_cflag_nostream );
   //this->addConstant( "C_RAW", (int64)FALCON_VFS_MODE_FLAG_RAW );
}

ModuleVFS::~ModuleVFS()
{}

}}

/* end of vfs_fm.cpp */

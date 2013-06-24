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

#undef SRC
#define SRC "falcon/moduels/native/feathers/vfs.cpp"

#include <falcon/symbol.h>
#include <falcon/item.h>
#include <falcon/vfsprovider.h>
#include <falcon/errors/linkerror.h>

#include "vfs.h"
#include "vfs_ext.h"
#include "classvfs.h"
#include "classfilestat.h"
#include "classdirectory.h"

namespace Falcon {

VFSModule::VFSModule():
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

void VFSModule::onLinkComplete()
{
   Item* i_uriClass = resolveGlobally("URI");
   Item* i_streamClass = resolveGlobally("Stream");

   if( i_uriClass == 0 || ! i_uriClass->isClass() )
   {
      throw new Falcon::LinkError( Falcon::ErrorParam( Falcon::e_link_error, __LINE__, SRC )
          .module(this->name())
         .extra( "Class URI not found" ) );
   }

   if( i_streamClass == 0 || ! i_streamClass->isClass() )
   {
      throw new Falcon::LinkError( Falcon::ErrorParam( Falcon::e_link_error, __LINE__, name() )
         .module(this->name())
         .extra( "Class Stream not found" ) );
   }

  m_uriClass = static_cast<Class*>(i_uriClass->asInst());
  m_streamClass = static_cast<Class*>(i_streamClass->asInst());
}

VFSModule::~VFSModule()
{}

}

//=================================
// Export function.

FALCON_MODULE_DECL 
{
   Falcon::Module* mod = new Falcon::VFSModule;
   return mod;
}

/* end of vfs.cpp */

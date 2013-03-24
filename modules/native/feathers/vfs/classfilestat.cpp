/*
   FALCON - The Falcon Programming Language.
   FILE: classfilestat.cpp

   Falcon core module -- Structure holding information on files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Mar 2013 00:25:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "modules/native/feathers/vfs/classfilestat.cpp"

#include <falcon/function.h>
#include <falcon/vmcontext.h>
#include <falcon/errors/paramerror.h>
#include <falcon/uri.h>
#include <falcon/cm/uri.h>
#include <falcon/filestat.h>
#include <falcon/stream.h>

#include "vfs.h"
#include "classfilestat.h"

namespace Falcon {
namespace Ext {

namespace _classFileStat {

}

//=========================================================
//
//=========================================================


ClassFileStat::ClassFileStat():
         Class("FileStat")
{
   addConstant( "NOT_FOUND", FileStat::_notFound );
   addConstant( "UNKNOWN", FileStat::_unknown );
   addConstant( "NORMAL", FileStat::_normal );
   addConstant( "DIR", FileStat::_dir );
   addConstant( "LINK", FileStat::_link );
   addConstant( "DEVICE", FileStat::_device );
   addConstant( "SOCKET", FileStat::_socket );
}


ClassFileStat::~ClassFileStat()
{
}

void ClassFileStat::dispose( void* ) const
{
   // FOR NOW, do nothing
}

void* ClassFileStat::clone( void* instance ) const
{
   FileStat* fs = static_cast<FileStat*>(instance);
   return new FileStat(*fs);
}

void* ClassFileStat::createInstance() const
{
   return new Falcon::FileStat;
}

}
}

/* end of classfilestat.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: pipe.cpp

   System independent abstraction for linked inter-process sockets.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/pipe.cpp"

#include <falcon/pipe.h>
#include <falcon/errors/ioerror.h>
#include <falcon/fstream.h>

namespace Falcon {
namespace Sys {

Pipe::~Pipe()
{
   close();
}

void Pipe::close()
{
   closeRead();
   closeWrite();
}


ReadOnlyFStream* Pipe::getReadStream()
{
   FileData* fd = new FileData();
   m_readSide.passOn( *fd );
   return new ReadOnlyFStream( fd );
}


WriteOnlyFStream* Pipe::getWriteStream()
{
   FileData* fd = new FileData();
   m_writeSide.passOn( *fd );
   return new WriteOnlyFStream( fd );
}

}
}

/* end of pipe.cpp */

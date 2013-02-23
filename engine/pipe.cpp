/*
   FALCON - The Falcon Programming Language.
   FILE: pipe.h

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

#include <unistd.h>
#include <errno.h>

namespace Falcon {
namespace Sys {

Pipe::Pipe()
{
   int fds[2];
   int result = pipe(fds);
   if( result != 0 )
   {
      throw new IOError( ErrorParam( e_io_open, __LINE__, SRC)
               .extra("pipe()")
               .sysError(errno));
   }

   m_readSide.fdFile = fds[0];
   m_writeSide.fdFile = fds[1];

   m_readSide.m_nonBloking = false;
   m_writeSide.m_nonBloking = false;
}


Pipe::~Pipe()
{
   close();
}

void Pipe::closeRead()
{
   if( m_readSide.fdFile != -1 )
   {
      int result = ::close( m_readSide.fdFile );
      m_readSide.fdFile = -1;

      if( result != 0 )
      {
         throw new IOError( ErrorParam( e_io_close, __LINE__, SRC)
                  .extra("close()")
                  .sysError(errno));
      }
   }
}

void Pipe::closeWrite()
{
   if( m_writeSide.fdFile != -1 )
   {
      int result = ::close( m_writeSide.fdFile );
      m_writeSide.fdFile = -1;

      if( result != 0 )
      {
         throw new IOError( ErrorParam( e_io_close, __LINE__, SRC)
                  .extra("close()")
                  .sysError(errno));
      }
   }
}

void Pipe::close()
{
   closeRead();
   closeWrite();
}

}
}

/* end of pipe.cpp */

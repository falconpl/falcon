/*
   FALCON - The Falcon Programming Language.
   FILE: pipe_win.cpp

   Windows specific Pipe functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai, Paul Davey
   Begin: Fri, 22 Feb 2013 20:01:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/pipe.cpp"

#include <falcon/pipe.h>
#include <falcon/errors/ioerror.h>

#include <windows.h>

namespace Falcon {
namespace Sys {

Pipe::Pipe()
{
   HANDLE write, read;
   SECURITY_ATTRIBUTES sec = {sizeof(SECURITY_ATTRIBUTES), NULL, true};
   BOOL result = CreatePipe(&read, &write, &sec, 0);
   if( result == 0 )
   {
      throw new IOError( ErrorParam( e_io_open, __LINE__, SRC)
               .extra("CreatePipe()")
               .sysError(GetLastError()));
   }

   m_readSide.hFile = read;
   m_writeSide.hFile = write;

   m_readSide.bNonBlocking = false;
   m_writeSide.bNonBlocking = false;
}


void Pipe::closeRead()
{
   if( m_readSide.hFile != NULL )
   {
      BOOL result = ::CloseHandle( m_readSide.hFile );
      m_readSide.hFile = NULL;

      if( result == 0 )
      {
         throw new IOError( ErrorParam( e_io_close, __LINE__, SRC)
                  .extra("CloseHandle()")
                  .sysError(GetLastError()));
      }
   }
}

void Pipe::closeWrite()
{
   if( m_writeSide.hFile != NULL )
   {
      BOOL result = ::CloseHandle( m_writeSide.hFile );
      m_writeSide.hFile = NULL;

      if( result == 0 )
      {
         throw new IOError( ErrorParam( e_io_close, __LINE__, SRC)
                  .extra("CloseHandle()")
                  .sysError(GetLastError()));
      }
   }
}

}
}

/* end of pipe.cpp */

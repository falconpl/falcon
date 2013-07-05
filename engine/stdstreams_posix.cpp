/*
   FALCON - The Falcon Programming Language
   FILE: stdstreams_posix.cpp

   Unix specific standard streams factories.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: ven ago 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix specific standard streams factories.
*/

#include <falcon/stdstreams.h>
#include <unistd.h>
#include <falcon/stderrors.h>

namespace Falcon {

inline int make_fd( int origfd, bool bDup )
{
   int fd;

   if ( bDup )
   {
      fd = ::dup( origfd );
      if( fd < 0 )
      {
         throw new IOError( ErrorParam(e_io_dup, __LINE__, __FILE__).sysError(errno) );
      }

   }
   else
   {
      fd = origfd;
   }

   return fd;
}


StdInStream::StdInStream( bool bDup ):
   ReadOnlyFStream( new Sys::FileData(make_fd(STDIN_FILENO, bDup )) )
{}

StdOutStream::StdOutStream( bool bDup ):
   WriteOnlyFStream( new Sys::FileData(make_fd(STDOUT_FILENO, bDup )) )
{}

StdErrStream::StdErrStream( bool bDup ):
   WriteOnlyFStream( new Sys::FileData(make_fd(STDERR_FILENO, bDup )) )
{}

}


/* end of stdstreams_posix.cpp */

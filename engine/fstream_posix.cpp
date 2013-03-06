/*
   FALCON - The Falcon Programming Language.
   FILE: fstream_sys_unix.cpp

   Unix system specific FILE service support
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom mar 12 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix system specific FILE service support.
*/

#include <unistd.h>
#include <errno.h>
#include <sys/poll.h>

#include <falcon/fstream.h>
#include <falcon/errors/ioerror.h>
#include <falcon/errors/interruptederror.h>
#include <falcon/errors/unsupportederror.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace Falcon {

FStream::FStream( Sys::FileData* data ):
   m_fsData(data)
{
   m_status = t_open;
}

FStream::FStream( const FStream &other ):
   Stream( other )
{
   Sys::FileData* data = static_cast<Sys::FileData*>(other.m_fsData);
   int fd = static_cast<Sys::FileData*>(other.m_fsData)->fdFile;
   int fd2 = ::dup( fd );
   if ( fd2 < 0 )
   {
      throw new IOError (ErrorParam(e_io_dup, __LINE__, __FILE__ ).sysError(errno) );
   }

   m_fsData = new Sys::FileData(fd2, data->m_nonBloking);
}


FStream::~FStream()
{
   close();
   delete static_cast<Sys::FileData*>(m_fsData);
}


bool FStream::close()
{
   int fd = static_cast<Sys::FileData*>(m_fsData)->fdFile;

   if ( m_status & Stream::t_open ) {
      if( ::close( fd ) < 0 ) {
         m_lastError = (size_t) errno;
         m_status = m_status | t_error;

         if( m_bShouldThrow )
         {
            throw new IOError( ErrorParam( e_io_close, __LINE__, __FILE__ ).sysError( errno ) );
         }

         return false;
      }
   }

   m_lastError = 0;
   m_status = m_status & (~ Stream::t_open);
   return true;
}


size_t FStream::read( void *buffer, size_t size )
{
   Sys::FileData* data = static_cast<Sys::FileData*>(m_fsData);
   int fd = data->fdFile;
   int result = ::read( fd, buffer, size );
   if ( result < 0 ) {
      m_lastError = (size_t) errno;
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_read, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return -1;
   }

   if ( result == 0 ) {
      m_status = m_status | Stream::t_eof;
   }

   m_lastError = 0;
   return result;
}

size_t FStream::write( const void *buffer, size_t size )
{
   Sys::FileData* data = static_cast<Sys::FileData*>(m_fsData);
   int fd = data->fdFile;

   int result = ::write( fd, buffer, size );
   if ( result < 0 ) {
      m_lastError = (size_t) errno;
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return false;
   }

   m_lastError = 0;
   return result;
}

off_t FStream::seek( off_t pos, e_whence whence )
{
   int fd = static_cast<Sys::FileData*>(m_fsData)->fdFile;

   int from;
   switch( whence ) {
      case ew_begin: from = SEEK_SET; break;
      case ew_cur: from = SEEK_CUR; break;
      case ew_end: from = SEEK_END; break;
      default:
         from = SEEK_SET;
         break;
   }

   pos = (off_t) ::lseek( fd, pos, from );
   if( pos < 0 ) {
      m_lastError = errno;
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_seek, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return -1;
   }
   else
      m_status = m_status & ~Stream::t_eof;

   m_lastError = 0;
   return pos;
}


off_t FStream::tell()
{
   int fd = static_cast<Sys::FileData*>(m_fsData)->fdFile;

   off_t pos = (off_t) ::lseek( fd, 0, SEEK_CUR );

   if( pos < 0 ) {
      m_lastError = (size_t) errno;
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_seek, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return -1;
   }

   m_lastError = 0;
   return pos;
}


bool FStream::truncate( off_t pos )
{
   int fd = static_cast<Sys::FileData*>(m_fsData)->fdFile;

   if ( pos < 0 ) {
      pos = tell();
      if ( pos < 0 )
         return false;
   }

   int32 res = ::ftruncate( fd, pos );
   if( res < 0 ) {
      m_lastError = (size_t) errno;
      m_status = m_status | t_error;

      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_write, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return false;
   }

   m_lastError = 0;
   return true;
}

}

/* end of fstream_sys_unix.cpp */

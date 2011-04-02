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
#include <falcon/interrupt.h>
#include <falcon/ioerror.h>
#include <falcon/interruptederror.h>
#include <falcon/unsupportederror.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace Falcon {

FStream::FStream( void* data ):
   m_fsData(data)
{
}

FStream::FStream( const FStream &other ):
   Stream( other )
{
   int fd = *(int*) m_fsData;
   int fd2 = ::dup( fd );
   if ( fd2 < 0 )
   {
      throw new IOError (ErrorParam(e_io_dup, __LINE__, __FILE__ ) );
   }

   m_fsData = new int[1];
   *((int*)m_fsData) = fd2;
}


FStream::~FStream()
{
   close();
   delete[] (int*) m_fsData;
}


bool FStream::close()
{
   int fd = *(int*) m_fsData;

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
   int fd = *(int*) m_fsData;

   size_t result = ::read( fd, buffer, size );
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
   int fd = *(int*) m_fsData;

   size_t result = ::write( fd, buffer, size );
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
   int fd = *(int*) m_fsData;

   int from;
   switch( whence ) {
      case ew_begin: from = SEEK_SET; break;
      case ew_cur: from = SEEK_CUR; break;
      case ew_end: from = SEEK_END; break;
      default:
         from = SEEK_SET;
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
   int fd = *(int*) m_fsData;

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
   int fd = *(int*) m_fsData;

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

size_t FStream::readAvailable( int32 msec )
{   
   /* Temporarily turned off because a darwin flaw

   struct pollfd poller;
   poller.fd = data->m_handle;
   poller.events = POLLIN | POLLPRI;

   if ( poll( &poller, 1, msec) == 1 ) {
      data->m_lastError = 0;
      if( (poller.revents & (POLLIN | POLLPRI | POLLHUP )) != 0 )
         return 1;
   }
   else {
      if ( errno == EINPROGRESS ) {
         data->m_lastError = 0;
         return 0;
      }
      data->m_lastError = errno;
      return -1;
   }

   return 0;
   */

   int fd = *(int*) m_fsData;
   struct timeval tv, *tvp;
   fd_set set;
   int last;

   FD_ZERO( &set );
   FD_SET( fd, &set );
   if( m_ptrIntr.assigned() )
   {
      int* pipe_fds = (int*) m_ptrIntr->sysData();
      last = pipe_fds[0];

      FD_SET( last, &set );
      if( last < fd )
         last = fd;
   }
   else
      last = fd;

   if ( msec >= 0 ) {
      tvp = &tv;
      tv.tv_sec = msec / 1000;
      tv.tv_usec = (msec % 1000 ) * 1000;
   }
   else
      tvp = 0;

   switch( select( last + 1, &set, 0, 0, tvp ) )
   {
      case 1:
      case 2:
         if ( m_ptrIntr.assigned() && FD_ISSET( ((int*) m_ptrIntr->sysData())[0], &set ) )
         {
            m_status = m_status | t_interrupted;
            m_ptrIntr->reset();
            if( m_bShouldThrow )
            {
               throw new InterruptedError( ErrorParam( e_interrupted, __LINE__, __FILE__ ) );
            }

            return -1;
         }

         return FD_ISSET( fd, &set ) ? 1 : 0;

      case -1:
         if( errno == EINPROGRESS ) {
            m_lastError = 0;
            return 0;
         }

         m_lastError = (size_t) errno;
         if( m_bShouldThrow )
         {
            throw new IOError( ErrorParam( e_io_ravail, __LINE__, __FILE__ ).sysError( errno ) );
         }

         return -1;
   }

   return 0;
}

size_t FStream::writeAvailable( int32 msec )
{
   int fd = *(int*) m_fsData;

   struct pollfd poller[2];
   int fds;

   poller[0].fd = fd;
   poller[0].events = POLLOUT;

   if ( m_ptrIntr.assigned() )
   {
      int* poll_fds = (int*) m_ptrIntr->sysData();

      fds = 2;
      poller[1].fd = poll_fds[0];
      poller[1].events = POLLIN;
   }
   else
      fds = 1;


   int res;
   while( ( res = poll( poller, fds, msec ) ) == EAGAIN );

   if ( res == 0 )
   {
      m_lastError = 0;
      if( m_ptrIntr.assigned()  && (poller[1].revents & POLLIN) != 0 )
      {
         m_ptrIntr->reset();
         m_status = m_status | t_interrupted;
         if( m_bShouldThrow )
         {
            throw new InterruptedError( ErrorParam( e_interrupted, __LINE__, __FILE__ ) );
         }

         return -1;
      }

      if( (poller[0].revents & ( POLLOUT | POLLHUP ) ) != 0 )
         return 1;
   }
   else {
      m_lastError = (size_t) errno;
      if( m_bShouldThrow )
      {
         throw new IOError( ErrorParam( e_io_wavail, __LINE__, __FILE__ ).sysError( errno ) );
      }

      return -1;
   }

   return 0;
}


}

/* end of fstream_sys_unix.cpp */

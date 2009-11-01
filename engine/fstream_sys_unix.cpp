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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <string.h>

#include <falcon/autocstring.h>
#include <falcon/vm_sys_posix.h>
#include <falcon/fstream_sys_unix.h>
#include <falcon/memory.h>

namespace Falcon {

FileSysData *UnixFileSysData::dup()
{
   int fd2 = ::dup( m_handle );
   if ( fd2 >= 0 )
   {
      UnixFileSysData *other = new UnixFileSysData( fd2, m_lastError );
      return other;
   }
   else
      return 0;
}


BaseFileStream::BaseFileStream( const BaseFileStream &other ):
   Stream( other )
{
   m_fsData = other.m_fsData->dup();
}

BaseFileStream::~BaseFileStream()
{
   close();
   delete m_fsData;
}

bool BaseFileStream::close()
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   if ( m_status & Stream::t_open ) {
      if( ::close( data->m_handle ) == -1 ) {
         data->m_lastError = errno;
         m_status = t_error;
         return false;
      }
   }

   data->m_lastError = 0;
   m_status = m_status & (~ Stream::t_open);
   return true;
}

int32 BaseFileStream::read( void *buffer, int32 size )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   int32 result = ::read( data->m_handle, buffer, size );
   if ( result < 0 ) {
      data->m_lastError = errno;
      m_status = Stream::t_error;
      m_status = t_error;
      return -1;
   }

   if ( result == 0 ) {
      m_status = m_status | Stream::t_eof;
   }

   data->m_lastError = 0;
   m_lastMoved = result;
   return result;
}

int32 BaseFileStream::write( const void *buffer, int32 size )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   int32 result = ::write( data->m_handle, buffer, size );
   if ( result < 0 ) {
      data->m_lastError = errno;
      m_status = Stream::t_error;
      return -1;
   }
   data->m_lastError = 0;
   m_lastMoved = result;
   return result;
}

bool BaseFileStream::put( uint32 chr )
{
   /** \TODO optimize */
   byte b = (byte) chr;
   return write( &b, 1 ) == 1;
}

bool BaseFileStream::get( uint32 &chr )
{
   /** \TODO optimize */

   if( popBuffer( chr ) )
      return true;

   byte b;
   if ( read( &b, 1 ) == 1 )
   {
      chr = (uint32) b;
      return true;
   }
   return false;
}

int64 BaseFileStream::seek( int64 pos, e_whence whence )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   int from;
   switch( whence ) {
      case ew_begin: from = SEEK_SET; break;
      case ew_cur: from = SEEK_CUR; break;
      case ew_end: from = SEEK_END; break;
      default:
         from = SEEK_SET;
   }

   pos = (int64) lseek( data->m_handle, pos, from );
   if( pos < 0 ) {
      data->m_lastError = errno;
      m_status = Stream::t_error;
      return -1;
   }
   else
      m_status = m_status & ~Stream::t_eof;

   setError( 0 );
   return pos;
}


int64 BaseFileStream::tell()
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   int64 pos = (int64) lseek( data->m_handle, 0, SEEK_CUR );

   if( pos < 0 ) {
      setError( errno );
      return -1;
   }

   setError( 0 );
   return pos;
}


bool BaseFileStream::truncate( int64 pos )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   if ( pos < 0 ) {
      pos = tell();
      if ( pos < 0 )
         return false;
   }

   int32 res = ftruncate( data->m_handle, pos );
   if( res < 0 ) {
      setError( errno );
      return false;
   }

   setError( 0 );
   return true;
}

bool BaseFileStream::errorDescription( ::Falcon::String &description ) const
{
   if ( Stream::errorDescription( description ) )
      return true;

   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
   if ( data->m_lastError == 0 )
      return false;

   const char *error;

   if ( data->m_lastError == -1 )
      error = "Out of memory";

   error = strerror( data->m_lastError );
   if( error == 0 )
   {
      return false;
   }

   description.bufferize( error );
   return true;
}

int64 BaseFileStream::lastError() const
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
   return (int64) data->m_lastError;
}

void BaseFileStream::setError( int64 errorCode )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
   data->m_lastError = errorCode;
   if ( errorCode != 0 )
      status( t_error );
   else
      status( status() & ~Stream::t_error );
}

bool BaseFileStream::writeString( const String &content, uint32 begin, uint32 end )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
   uint32 done = begin;
   uint32 stop = content.size();
   uint32 charSize = content.manipulator()->charSize();
   if ( end < stop / charSize )
      stop = end * charSize;

   while ( done < stop )
   {
      int32 written = ::write( data->m_handle,
         (const char *) (content.getRawStorage() + done),
         content.size() - done );

      if ( written < 0 )
      {
         setError( errno );
         m_lastMoved = done;
         return false;
      }
      done += written;
   }

   setError( 0 );
   m_lastMoved = done - begin;
   return true;
}

bool BaseFileStream::readString( String &content, uint32 size )
{
   // TODO OPTIMIZE
   uint32 chr;
   content.size( 0 );
   //UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );

   while( size > 0 && get( chr ) )
   {
      size--;
      content.append( chr );
   }

   if ( size == 0 || eof() )
   {
      setError( 0 );
      return true;
   }

   setError( errno );
   return false;
}


int32 BaseFileStream::readAvailable( int32 msec, const Sys::SystemData *sysData )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
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

   struct timeval tv, *tvp;
   fd_set set;
   int last;

   FD_ZERO( &set );
   FD_SET( data->m_handle, &set );
   if( sysData != 0 )
   {
      last = sysData->m_sysData->interruptPipe[0];
      FD_SET( last, &set );
      if( last < data->m_handle )
         last = data->m_handle;
   }
   else
      last = data->m_handle;

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
         if ( sysData != 0 && FD_ISSET( sysData->m_sysData->interruptPipe[0], &set ) )
         {
            m_status = t_interrupted;
            return -1;
         }

      return FD_ISSET( data->m_handle, &set ) ? 1 : 0;

      case -1:
         if( errno == EINPROGRESS ) {
            setError( 0 );
            return 0;
         }
         setError( errno );
      return -1;
   }

   return 0;
}

int32 BaseFileStream::writeAvailable( int32 msec, const Sys::SystemData *sysData )
{
   UnixFileSysData *data = static_cast< UnixFileSysData *>( m_fsData );
   struct pollfd poller[2];
   int fds;

   poller[0].fd = data->m_handle;
   poller[0].events = POLLOUT;

   if ( sysData != 0 )
   {
      fds = 2;
      poller[1].fd = sysData->m_sysData->interruptPipe[0];
      poller[1].events = POLLIN;
   }
   else
      fds = 1;


   int res;
   while( ( res = poll( poller, fds, msec ) ) == EAGAIN );

   if ( res > 0 )
   {
      setError( 0 );
      if( sysData != 0 && (poller[1].revents & POLLIN) != 0 )
      {
         m_status = t_interrupted;
         return -1;
      }

      if( (poller[0].revents & ( POLLOUT | POLLHUP ) ) != 0 )
         return 1;
   }
   else {
      setError( errno );
      return -1;
   }

   return 0;
}

BaseFileStream *BaseFileStream::clone() const
{
   BaseFileStream *ge = new BaseFileStream( *this );
   if ( ge->m_fsData == 0 )
   {
      delete ge;
      return 0;
   }

   return ge;
}

//=========================================
// File Stream
//=========================================

FileStream::FileStream():
   BaseFileStream( t_file, new UnixFileSysData( -1, 0 ) )
{
   status( t_none );
}

bool FileStream::open( const String &filename, t_openMode mode, t_shareMode share )
{
   UnixFileSysData *data = static_cast< UnixFileSysData * >( m_fsData );
   int omode = 0;

   if ( data->m_handle > 0 )
      ::close( data->m_handle );

   if ( mode == e_omReadWrite )
      omode = O_RDWR;
   else if ( mode == e_omReadOnly )
      omode = O_RDONLY;
   else
      omode = O_WRONLY;

   // todo: do something about share mode
   AutoCString cfilename( filename );

   int handle;
   errno = 0;
   handle = ::open( cfilename.c_str(), omode );

   data->m_handle = handle;
   if ( handle < 0 ) {
      setError( errno );
      status( t_error );
      return false;
   }

   status( t_open );
   data->m_lastError = 0;
   return true;
}

bool FileStream::create( const String &filename, t_attributes mode, t_shareMode share )
{
   UnixFileSysData *data = static_cast< UnixFileSysData * >( m_fsData );

   if ( data->m_handle > 0 )
      ::close( data->m_handle );

   //TODO: something about sharing
   AutoCString cfilename( filename );
   errno=0;
   data->m_handle = ::open( cfilename.c_str(), O_CREAT | O_RDWR | O_TRUNC, static_cast<uint32>( mode ) );

   if ( data->m_handle < 0 ) {
      data->m_lastError = errno;
      status( t_error );
      return false;
   }

   status( t_open );
   data->m_lastError = 0;
   return true;
}

void FileStream::setSystemData( const FileSysData &fsData )
{
   const UnixFileSysData *data = static_cast< const UnixFileSysData *>( &fsData );
   UnixFileSysData *myData = static_cast< UnixFileSysData *>( m_fsData );
   myData->m_handle = data->m_handle;
   myData->m_lastError = data->m_lastError;
}

StdInStream::StdInStream():
   InputStream( new UnixFileSysData( dup(STDIN_FILENO), 0 ) )
{
}

StdOutStream::StdOutStream():
   OutputStream( new UnixFileSysData( dup(STDOUT_FILENO), 0 ) )
{
}

StdErrStream::StdErrStream():
   OutputStream( new UnixFileSysData( dup(STDERR_FILENO), 0 ) )
{
}

RawStdInStream::RawStdInStream():
   InputStream( new UnixFileSysData( STDIN_FILENO, 0 ) )
{
}

RawStdOutStream::RawStdOutStream():
   OutputStream( new UnixFileSysData( STDOUT_FILENO, 0 ) )
{
}

RawStdErrStream::RawStdErrStream():
   OutputStream( new UnixFileSysData( STDERR_FILENO, 0 ) )
{
}


}

/* end of fstream_sys_unix.cpp */

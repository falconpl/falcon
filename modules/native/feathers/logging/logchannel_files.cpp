/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_files.cpp

   Logging module -- log channel interface (for self-rolling files)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel_files.cpp"

#include "logchannel_files.h"
#include <falcon/engine.h>
#include <falcon/vfsiface.h>
#include <falcon/stream.h>
#include <falcon/textwriter.h>
#include <falcon/tc/transcoderutf8.h>

namespace Falcon {
namespace Feathers {

//==========================================================
// Stream writing to a rotor file
//==========================================================

LogChannelFiles::LogChannelFiles( const String& path, int level ):
   LogChannel( level ),
   m_stream(0),
   m_bFlushAll( false ),
   m_path( path ),
   m_maxSize( 0 ),
   m_maxCount( 0 ),
   m_bOverwrite( 0 ),
   m_maxDays( 0 ),
   m_isOpen( false )
{
   m_encoding = FALCON_TRANSCODER_UTF8_NAME;
}


LogChannelFiles::LogChannelFiles( const String& path, const String &fmt, int level ):
   LogChannel( fmt, level ),
   m_stream(0),
   m_bFlushAll( false ),
   m_path( path ),
   m_maxSize( 0 ),
   m_maxCount( 0 ),
   m_bOverwrite( 0 ),
   m_maxDays( 0 ),
   m_isOpen( false )
{
   m_encoding = FALCON_TRANSCODER_UTF8_NAME;
}


LogChannelFiles::~LogChannelFiles()
{
   close();
   if( m_stream != 0 )
   {
      m_stream->decref();
   }
}


void LogChannelFiles::log( const String& tgt, const String& source, const String& function, uint32 level, const String& msg, uint32 code )
{
   open();
   LogChannel::log( tgt, source, function, level, msg, code );
}

void LogChannelFiles::open()
{
   m_mtx_open.lock();
   if( m_isOpen )
   {
      m_mtx_open.unlock();
      return;
   }
   else {
      m_isOpen = true;
   }
   m_mtx_open.unlock();

   VFSProvider* vfs = Engine::instance()->vfs().getVFS("");
   Stream* stream;

   String fname;
   expandPath( 0, fname );

   m_opendate.currentTime();

   try
   {
      if ( ! m_bOverwrite )
      {
         stream = vfs->open( fname, VFSIface::OParams(VFSIface::OParams::e_oflag_append | VFSIface::OParams::e_oflag_wr) );
      }
      else
      {
         stream = vfs->create( fname, VFSIface::CParams(VFSIface::CParams::e_oflag_wr) );
      }

      Transcoder* tc = Engine::instance()->getTranscoder(m_encoding);
      fassert( tc != 0 ); // already checked elsewhere


      // change the stream thread-wise
      TextWriter* nstream = new TextWriter(stream, tc);
      m_mtx_open.lock();
      TextWriter* oldStream = m_stream;
      m_stream = nstream;
      m_mtx_open.unlock();

      // overkill...
      if( oldStream != 0 )
      {
         oldStream->decref();
      }
   }
   catch( ... )
   {
      m_mtx_open.lock();
      m_isOpen = false;
      m_mtx_open.unlock();
      throw;
   }
}


void LogChannelFiles::expandPath( int32 number, String& path )
{
   path = m_path;

   uint32 pos = path.find( "%" );

   String temp;
   if ( m_maxCount == 0 )
   {
      // expand "%" into ""
      temp = "";
   }
   else
   {
      temp.writeNumber((int64) number);
      uint32 count;
      if ( m_maxCount > 100000000 )
         count = 9;
      else if ( m_maxCount > 10000000 )
         count = 8;
      else if( m_maxCount > 1000000 )
         count = 7;
      else if( m_maxCount > 100000 )
         count = 6;
      else if( m_maxCount > 10000 )
         count = 5;
      else if( m_maxCount > 1000 )
         count = 4;
      else if( m_maxCount > 100 )
         count = 3;
      else if( m_maxCount > 10 )
         count = 2;
      else
         count = 1;

      while( temp.length() < count )
      {
         temp.prepend(0);
      }
   }

   // change, or eventually append the number
   if ( pos != String::npos )
   {
      path.change( pos, pos+1, temp );
   }
   else {
      path.append( "." );
      path.append( temp );
   }

}


void LogChannelFiles::reset()
{
   pushBack( new LogMessage( "", "", ".", 0, "", 0 ) );

}


void LogChannelFiles::rotate()
{
   pushBack( new LogMessage( "", "", ".", 0, "", 1 ) );
}


void LogChannelFiles::writeLogEntry( const String& entry, LogMessage* pOrigMsg )
{
   // special management
   // if the source of the message is ".", then we have a special order
   if( pOrigMsg->m_caller == "." )
   {
      // roll?
      if ( pOrigMsg->m_code == 1 )
      {
         // if flushing all, we don't need to do it now.
         // But the setting may change across the threads...
         // so better stay on the bright side.
         m_stream->flush();
         inner_rotate();
      }
      else
      {
         m_stream->underlying()->truncate(0);
      }

      return;
   }

   // for now, just write.
   // TODO roll files.
   m_stream->writeLine( entry );

   if( m_maxSize > 0 && m_stream->underlying()->tell() > m_maxSize )
   {
      m_stream->flush();
      inner_rotate();
   }
   else if( m_maxDays > 0 )
   {
      TimeStamp maxDate = m_opendate;
      maxDate.add( m_maxDays );
      // are we greater than the last calculated timestamp?
      if( maxDate.compare( m_ts ) > 0 )
      {
         m_stream->flush();
         inner_rotate();
         m_opendate.currentTime();
      }
   }
   else if ( m_bFlushAll )
   {
      m_stream->flush();
   }
}


void LogChannelFiles::inner_rotate()
{
   if ( m_maxCount > 0 )
   {
      VFSProvider* vfs = Engine::instance()->vfs().getVFS("");

     // find the maximum file
      int maxNum;
      for( maxNum = 1; maxNum <= m_maxCount; maxNum++ )
      {
         String fname;
         expandPath( maxNum, fname );

         FileStat ft;
         if( ! vfs->readStats( fname, ft, true ) )
         {
            break;
         }
      }

      while( maxNum > 0 )
      {
         String from, into;
         expandPath( maxNum, from );
         expandPath( --maxNum, into );
         vfs->move( from, into );
      }

      m_mtx_open.lock();
      m_isOpen = false;
      m_mtx_open.unlock();
      open();
   }
   else
   {
      m_stream->underlying()->truncate(0);
   }
}


String LogChannelFiles::encoding() const
{
   m_mtx_open.lock();
   String temp = m_encoding;
   m_mtx_open.unlock();
   return temp;
}

bool LogChannelFiles::encoding( const String& enc )
{
   if( Engine::instance()->getTranscoder(enc) == 0 )
   {
      return false;
   }

   m_mtx_open.lock();
   m_encoding = enc;
   m_mtx_open.unlock();
   return true;
}


bool LogChannelFiles::close()
{
   if( ! LogChannel::close() )
   {
      return false;
   }

   m_stream->flush();
   m_stream->underlying()->close();
   m_stream->decref();
   m_stream = 0;

   return true;
}

}
}

/* end of logchannel_files.cpp */

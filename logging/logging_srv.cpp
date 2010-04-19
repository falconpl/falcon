/*
   FALCON - The Falcon Programming Language
   FILE: logging_srv.cpp

   Logging module -- service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Sep 2009 16:42:41 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/srv/logging_srv.h>
#include <falcon/stream.h>
#include <falcon/fstream.h>
#include <falcon/error.h>
#include <falcon/time_sys.h>
#include <falcon/sys.h>
#include <falcon/filestat.h>
#include <falcon/mt.h>

namespace Falcon {

LogChannel::LogChannel( uint32 l ):
   m_refCount( 1 ),
   m_msg_head(0),
   m_msg_tail(0),
   m_terminate(false),
   m_level( l )
   {
      m_startedAt = Sys::Time::seconds();
      start();
   }

LogChannel::LogChannel( const String &format, uint32 l ):
   m_refCount( 1 ),
   m_msg_head(0),
   m_msg_tail(0),
   m_terminate(false),
   m_level( l ),
   m_format(format)
   {
      m_startedAt = Sys::Time::seconds();
      start();
   }


LogChannel::~LogChannel()
{
   stop();

   while( m_msg_head !=0 )
   {
      LogMessage* lm = m_msg_head;
      m_msg_head = m_msg_head->m_next;
      delete lm;
   }
}

void LogChannel::start()
{
   m_thread = new SysThread(this);
   m_thread->start( ThreadParams().stackSize(0xA000) );
}

void LogChannel::stop()
{
   if ( m_thread !=0 )
   {
      m_msg_mtx.lock();
      m_terminate = true;
      m_message_incoming.set();
      m_msg_mtx.unlock();

      void* res;
      m_thread->join( res );
      m_thread =  0;
   }
}


void LogChannel::incref()
{
   atomicInc( m_refCount );
}

void LogChannel::decref()
{
   if ( atomicDec(m_refCount) == 0 )
   {
      delete this;
   }
}

void LogChannel::log( LogArea* area, uint32 level, const String& msg, uint32 code )
{
   log( area->name(), "", level, msg, code );
}

void LogChannel::log( const String& area, const String& mod, const String& func, uint32 l, const String& msg, uint32 code )
{
   if ( l <= m_level )
   {
      // delegate formatting to the other thread.
      LogMessage* lmsg = new LogMessage( area, mod, func, l, msg, code );

      m_msg_mtx.lock();
      if ( m_terminate)
      {
         delete lmsg;
         m_msg_mtx.unlock();
         return;
      }

      if ( m_msg_tail == 0 )
      {
         m_msg_head = m_msg_tail = lmsg;
      }
      else
      {
         m_msg_tail->m_next = lmsg;
         m_msg_tail = lmsg;
      }
      m_msg_mtx.unlock();
      m_message_incoming.set();
   }
}


void LogChannel::pushFront( LogMessage* lmsg )
{
   m_msg_mtx.lock();
   if ( m_msg_tail == 0 )
   {
      m_msg_head = m_msg_tail = lmsg;
   }
   else
   {
      lmsg->m_next = m_msg_head;
      m_msg_head = lmsg;
   }
   m_msg_mtx.unlock();
   m_message_incoming.set();
}


void LogChannel::pushBack( LogMessage* lmsg )
{
   m_msg_mtx.lock();
   if ( m_msg_tail == 0 )
   {
      m_msg_head = m_msg_tail = lmsg;
   }
   else
   {
      m_msg_tail->m_next = lmsg;
      m_msg_tail = lmsg;
   }
   m_msg_mtx.unlock();
   m_message_incoming.set();
}



void* LogChannel::run()
{
   while( true )
   {
      m_message_incoming.wait(-1);
      m_msg_mtx.lock();
      if( m_terminate )
      {
         m_msg_mtx.unlock();
         break;
      }

      // copy the format for multiple usage
      String fmt = m_format;
      m_bTsReady = false;
      while( m_msg_head !=0 )
      {
         LogMessage* msg = m_msg_head;
         m_msg_head = m_msg_head->m_next;
         if ( m_msg_head == 0 )
            m_msg_tail = 0;
         m_msg_mtx.unlock();

         String target;
         if( expandMessage( msg, fmt, target ) )
         {
            writeLogEntry( target, msg );
         }
         else
         {
            writeLogEntry( msg->m_msg, msg );
         }

         delete msg;
         m_msg_mtx.lock();
      }

      m_msg_mtx.unlock();
   }

   return 0;
}


void LogChannel::setFormat( const String& fmt )
{
   m_msg_mtx.lock();
   m_format = fmt;
   m_msg_mtx.unlock();
}


void LogChannel::getFormat( String& fmt )
{
   m_msg_mtx.lock();
   fmt = m_format;
   m_msg_mtx.unlock();
}


bool LogChannel::expandMessage( LogMessage* msg, const String& fmt, String& target )
{

   if ( fmt == "" || fmt == "%m" )
      return false;

   target = fmt;
   uint32 pos = target.find( "%" );
   numeric distance;

   while( pos != String::npos )
   {
      String temp;

      if( pos + 1 == target.length() )
      {
         target.change(pos,"<?>");
         return true;
      }

      uint32 chr = target.getCharAt( pos + 1 );
      switch( chr )
      {
      case 't':
         updateTS();
         m_ts.toString( temp );
         target.change( pos, pos + 2, temp.subString(11) );
         break;

      case 'T':
         updateTS();
         m_ts.toString( temp );
         target.change( pos, pos + 2, temp );
         break;

      case 'd':
         updateTS();
         m_ts.toString( temp );
         target.change( pos, pos + 2, temp.subString(0,10) );
         break;

      case 'R':
         updateTS();
         m_ts.toRFC2822( temp );
         target.change( pos, pos + 2, temp );
         break;

      case 'S':
         distance = Sys::Time::seconds() - m_startedAt;
         temp.writeNumber( distance, "%.3f" );
         target.change( pos, pos + 2, temp );
         break;

      case 's':
         distance = Sys::Time::seconds() - m_startedAt;
         temp.writeNumber( (int64) (distance*1000), "%d" );
         target.change( pos, pos + 2, temp );
         break;

      case 'c':
       temp.writeNumber( (int64) msg->m_code );
         target.change( pos, pos + 2, temp );
         break;

      case 'C':
       temp.writeNumber( (int64) msg->m_code );
       {
       for( int i = temp.length(); i < 5; i ++ )
          temp.prepend( '0' );
       }
         target.change( pos, pos + 2, temp );
         break;

      case 'a':
         target.change( pos, pos + 2, msg->m_areaName );
         pos += msg->m_areaName.length();
         break;

      case 'M':
         target.change( pos, pos + 2, msg->m_modName );
         pos += msg->m_modName.length();
         break;

      case 'f':
         target.change( pos, pos + 2, msg->m_caller );
         pos += msg->m_caller.length();
         break;

      case 'm':
         target.change( pos, pos + 2, msg->m_msg );
         break;

      case 'l':
         temp.writeNumber( (int64) msg->m_level );
         target.change( pos, pos + 2, temp );
         break;

      case 'L':
         switch( msg->m_level )
         {
            case LOGLEVEL_FATAL: temp = "L"; break;
            case LOGLEVEL_ERROR: temp = "E"; break;
            case LOGLEVEL_WARN: temp = "W"; break;
            case LOGLEVEL_INFO: temp = "I"; break;
            case LOGLEVEL_DEBUG: temp = "D"; break;
            default: temp = "l";
         }

         target.change( pos, pos + 2, temp );
         break;

      case '%':
         target.change( pos, pos + 2, "%" );
         ++pos;
         break;

      /*
       *  TODO
       - %xy: Current year, 4 characters.
      - %xY: Current year, 2 characters.
      - %xM: current month, 2 characters.
      - %xd: Current day of month.
      - %xh: Current hour, 24 hours format
      - %xh: Current hour, 12 hours with a/p attached (3 characters).
      - %xm: current minute, 2 characters.
      - %xs: Current second, 2 characters.
      - %xS: Current millisecond, 3 characters.
      case 'x':
         // we have another character.
         if( pos + 2 >= target.length() )
         {
            target.change( pos , "<?>" );
            return true;
         }

         chr = target.getCharAt( pos + 2 );
         switch( chr )
         {
            case 'y': updateTS(); temp.write( m_ts.m_year ); break;
         }
         */
      }

      pos = target.find( "%", pos + 1 );

   }

   return true;
}

//==========================================================
// Log Area
//==========================================================

LogArea::~LogArea()
{
   m_mtx_chan.lock();
   while( m_head_chan != 0 )
   {
      ChannelCarrier* cc = m_head_chan;
      m_head_chan = m_head_chan->m_next;
      cc->m_channel->decref();
      delete cc;
   }
   m_mtx_chan.unlock();
}


void LogArea::log( uint32 level, const String& source, const String& func, const String& msg, uint32 code ) const
{
   m_mtx_chan.lock();
   ChannelCarrier* cc = m_head_chan;
   while( cc != 0 )
   {
      cc->m_channel->log( this->name(), source, func, level, msg, code );
      cc = cc->m_next;
   }
   m_mtx_chan.unlock();
}

int LogArea::minlog() const
{
   int ml = -1;
   m_mtx_chan.lock();
   ChannelCarrier* cc = m_head_chan;
   while( cc != 0 )
   {
      if ( ml < (int) cc->m_channel->level() )
         ml = cc->m_channel->level();
      cc = cc->m_next;
   }
   m_mtx_chan.unlock();

   return ml;
}

void LogArea::incref()
{
   atomicInc( m_refCount );
}


void LogArea::decref()
{
   if( atomicDec( m_refCount ) == 0 )
      delete this;
}

void LogArea::addChannel( LogChannel* chn )
{
   chn->incref();
   ChannelCarrier* cc = new ChannelCarrier( chn );
   cc->m_prev = 0;

   m_mtx_chan.lock();
   cc->m_next = m_head_chan;
   if( m_head_chan == 0 )
   {
      m_head_chan = cc;
   }
   else {
      m_head_chan->m_prev = cc;
      m_head_chan = cc;
   }
   m_mtx_chan.unlock();
}


void LogArea::removeChannel( LogChannel* chn )
{
   m_mtx_chan.lock();
   ChannelCarrier* cc = m_head_chan;
   while( cc != 0 )
   {
      if ( chn == cc->m_channel )
      {
         if ( cc->m_prev != 0 )
         {
            cc->m_prev->m_next = cc->m_next;
         }
         else
         {
            m_head_chan = cc->m_next;
         }

         if ( cc->m_next != 0 )
         {
            cc->m_next->m_prev = cc->m_prev;
         }
         chn->decref();
         delete cc;
         break;
      }
   }
   m_mtx_chan.unlock();
}

//==========================================================
// Stream writing to a stream channel
//==========================================================

LogChannelStream::LogChannelStream( Stream* s, int level ):
   LogChannel( level ),
   m_stream( s ),
   m_bFlushAll( true )
   {}

LogChannelStream::LogChannelStream( Stream* s, const String &fmt, int level ):
   LogChannel( fmt, level ),
   m_stream( s ),
   m_bFlushAll( true )
   {}

void LogChannelStream::writeLogEntry( const String& entry, LogChannel::LogMessage* )
{
   m_stream->writeString(entry);
   m_stream->writeString("\n");

   if( m_bFlushAll )
      m_stream->flush();
}

LogChannelStream::~LogChannelStream()
{
   stop();
   delete m_stream;
}

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

}


LogChannelFiles::~LogChannelFiles()
{
   stop();
   delete m_stream;
}


void LogChannelFiles::log( const String& tgt, const String& source, const String& function, uint32 level, const String& msg, uint32 code )
{
   if ( ! m_isOpen )
   {
      m_isOpen = true;
      open();
   }

   LogChannel::log( tgt, source, function, level, msg, code );
}

void LogChannelFiles::open()
{
   delete m_stream; // just incase it was already open and we want to reopen it.
   m_stream = new FileStream();

   String fname;
   expandPath( 0, fname );

   m_opendate.currentTime();

   if ( ! m_bOverwrite )
   {
      // can we open it?
      if( m_stream->open( fname, FileStream::e_omReadWrite ) )
         return;
   }

   // ok try to create
   if( ! m_stream->create( fname,
         FileStream::e_aUserWrite | FileStream::e_aUserRead | FileStream::e_aGroupRead | FileStream::e_aOtherRead,
         FileStream::e_smShareRead ) )
   {
      throw new IoError( ErrorParam( e_file_output, __LINE__ )
            .origin( e_orig_runtime )
            .extra( fname )
            .sysError( (int32) m_stream->lastError() ) );
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
         m_stream->truncate(0);
      }

      return;
   }

   // for now, just write.
   // TODO roll files.
   m_stream->writeString( entry );
   m_stream->writeString( "\n" );

   if( m_maxSize > 0 && m_stream->tell() > m_maxSize )
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
      m_stream->close();
      delete m_stream;

      // find the maximum file
      int maxNum;
      for( maxNum = 1; maxNum <= m_maxCount; maxNum++ )
      {
         FileStat::e_fileType ft;
         String fname;
         expandPath( maxNum, fname );

         if( ! Sys::fal_fileType( fname, ft ) )
            break;
      }

      while( maxNum > 0 )
      {
         String from, into;
         expandPath( maxNum, from );
         expandPath( --maxNum, into );

         int32 fsStatus;
         Sys::fal_move( from, into, fsStatus );
      }

      String fname;
      expandPath( 0, fname );

      m_stream = new FileStream;
      // TODO? -- signal an error to the other thread?
      m_stream->create( fname,
            FileStream::e_aUserWrite | FileStream::e_aUserRead | FileStream::e_aGroupRead | FileStream::e_aOtherRead,
            FileStream::e_smShareRead );
   }
   else
   {
      m_stream->truncate(0);
   }
}


LogService::LogService():
   Service( LOGSERVICE_NAME )
{
}

LogArea* LogService::makeLogArea( const String& name ) const
{
   return new LogArea( name );
}

LogChannelStream* LogService::makeChnStream( Stream* s, int level ) const
{
   return new LogChannelStream( s, level );
}

LogChannelStream* LogService::makeChnStream( Stream* s, const String &fmt, int level ) const
{
   return new LogChannelStream( s, fmt, level );
}

LogChannelSyslog* LogService::makeChnSyslog( const String& identity, uint32 facility, int level ) const
{
   return new LogChannelSyslog( identity, facility, level );
}

LogChannelSyslog* LogService::makeChnSyslog( const String& identity, const String &fmt, uint32 facility, int level ) const
{
   return new LogChannelSyslog( identity, fmt, facility, level );
}

LogChannelFiles* LogService::makeChnlFiles( const String& path, int level ) const
{
   return new LogChannelFiles( path, level );
}

LogChannelFiles* LogService::makeChnFiles( const String& path, const String &fmt, int level ) const
{
   return new LogChannelFiles( path, fmt, level );
}


}

/* end of logging_srv.cpp */

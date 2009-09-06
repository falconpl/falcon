/*
   FALCON - The Falcon Programming Language
   FILE: logging_mod.cpp

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "logging_mod.h"
#include <falcon/stream.h>
#include <falcon/error.h>
#include <falcon/time_sys.h>

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
   m_msg_mtx.lock();
   m_terminate = true;
   m_message_incoming.set();
   m_msg_mtx.unlock();

   void* res;
   m_thread->join( res );
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

void LogChannel::log( LogArea* area, uint32 level, const String& msg )
{
   log( area->name(), "", level, msg );
}

void LogChannel::log( const String& area, const String& mod, const String& func, uint32 l, const String& msg )
{
   if ( l <= m_level )
   {
      // delegate formatting to the other thread.
      LogMessage* lmsg = new LogMessage( area, mod, func, l, msg );

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
}


void* LogChannel::run()
{
   while( true )
   {
      m_message_incoming.wait(-1);
      m_msg_mtx.lock();
      if( m_terminate )
         break;

      // copy the format for multiple usage
      String fmt = m_format;
      m_bTsReady = false;
      while( m_msg_head !=0 )
      {
         LogMessage* msg = m_msg_head;
         m_msg_head = m_msg_head->m_next;
         m_msg_mtx.unlock();

         String target;
         if( expandMessage( msg, fmt, target ) )
         {
            writeLogEntry( target );
         }
         else
         {
            writeLogEntry( msg->m_msg );
         }

         delete msg;
         m_msg_mtx.lock();
      }

      // we emptied the queue.
      m_msg_tail =0;
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
         temp.writeNumber( distance, "%.3g" );
         target.change( pos, pos + 2, temp );
         break;

      case 's':
         distance = Sys::Time::seconds() - m_startedAt;
         temp.writeNumber( (int64) (distance*1000) );
         target.change( pos, pos + 2, temp );
         break;

      case 'a':
         target.change( pos, pos + 2, msg->m_areaName );
         break;

      case 'M':
         target.change( pos, pos + 2, msg->m_modName );
         break;

      case 'f':
         target.change( pos, pos + 2, msg->m_caller );
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

      pos = target.find( "%", pos );

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


void LogArea::log( uint32 level, const String& source, const String& func, const String& msg )
{
   m_mtx_chan.lock();
   ChannelCarrier* cc = m_head_chan;
   while( cc != 0 )
   {
      cc->m_channel->log( this->name(), source, func, level, msg );
      cc = cc->m_next;
   }
   m_mtx_chan.unlock();
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
// Stream writing to a channel
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

void LogChannelStream::writeLogEntry( const String& entry )
{
   m_stream->writeString(entry);
   m_stream->writeString("\n");

   if( m_bFlushAll )
      m_stream->flush();
}

LogChannelStream::~LogChannelStream()
{
   delete m_stream;
}

}

/* end of logging_mod.cpp */

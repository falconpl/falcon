/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel.h

   Logging module -- log channel interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "falcon/modules/native/feathers/logging/logchannel.cpp"
#include "logchannel.h"
#include "logarea.h"

#include <falcon/sys.h>

namespace Falcon {
namespace Feathers {

LogChannel::LogChannel( uint32 l ):
   m_level( l ),
   m_msg_head(0),
   m_msg_tail(0),
   m_pool(0),
   m_poolSize(0),
   m_terminate(false),
   m_bClosed(false)
{
   m_startedAt = Sys::_seconds();
   start();
}

LogChannel::LogChannel( const String &format, uint32 l ):
   m_level( l ),
   m_format(format),
   m_msg_head(0),
   m_msg_tail(0),
   m_pool(0),
   m_poolSize(0),
   m_terminate(false),
   m_bClosed(false)
{
   m_startedAt = Sys::_seconds();
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
      if( m_terminate )
      {
         m_msg_mtx.unlock();
         return;
      }

      m_terminate = true;
      m_message_incoming.set();
      m_msg_mtx.unlock();

      void* res;
      m_thread->join( res );
      m_thread =  0;
   }
}

void LogChannel::log( LogArea* area, uint32 level, const String& msg, uint32 code )
{
   log( area->name(), "", level, msg, code );
}

void LogChannel::log( const String& area, const String& mod, const String& func, uint32 l, const String& msg, uint32 code )
{
   // first, perform an optimistic check
   if ( l <= m_level && ! m_bClosed )
   {
      // delegate formatting to the other thread.
      LogMessage* lmsg = allocMessage( area, mod, func, l, msg, code );

      m_msg_mtx.lock();
      // recheck level now.
      if ( m_terminate || m_bClosed || l > m_level )
      {
         m_msg_mtx.unlock();
         disposeMessage( lmsg );
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
         {
            m_msg_tail = 0;
         }

         if( m_bClosed )
         {
            m_msg_mtx.unlock();
            disposeMessage( msg );
            m_msg_mtx.lock();
            continue;
         }
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

         disposeMessage( msg );
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
         distance = Sys::_seconds() - m_startedAt;
         temp.writeNumber( distance, "%.3f" );
         target.change( pos, pos + 2, temp );
         break;

      case 's':
         distance = Sys::_seconds() - m_startedAt;
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
         pos += msg->m_msg.length();
         break;

      case 'l':
         temp.writeNumber( (int64) msg->m_level );
         target.change( pos, pos + 2, temp );
         pos += temp.length();
         break;

      case 'L':
         switch( msg->m_level )
         {
            case LOGLEVEL_FATAL: temp = "L"; break;
            case LOGLEVEL_ERROR: temp = "E"; break;
            case LOGLEVEL_WARN: temp = "W"; break;
            case LOGLEVEL_INFO: temp = "I"; break;
            case LOGLEVEL_DEBUG: temp = "D"; break;
            default: temp = "l"; break;
         }

         target.change( pos, pos + 2, temp );
         pos += temp.length();
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

bool LogChannel::close()
{
   m_msg_mtx.lock();
   if( m_bClosed )
   {
      m_msg_mtx.unlock();
      return false;
   }

   m_bClosed = true;
   m_msg_mtx.unlock();

   return true;
}

bool LogChannel::closed() const
{
   m_msg_mtx.lock();
   bool cl = m_bClosed;
   m_msg_mtx.unlock();
   return cl;
}


LogChannel::LogMessage* LogChannel::allocMessage(
         const String& areaName, const String& modname, const String& caller, int level, const String& msg, uint32 code )
{
   LogMessage* logmsg = 0;

   m_mtx_pool.lock();
   if( m_poolSize > 0 )
   {
      fassert( m_pool != 0 );
      logmsg = m_pool;
      m_pool = m_pool->m_next;
      m_poolSize--;
      m_mtx_pool.unlock();

      logmsg->m_areaName = areaName;
      logmsg->m_modName = modname;
      logmsg->m_caller = caller;
      logmsg->m_level = level;
      logmsg->m_msg = msg;
      logmsg->m_code = code;
   }
   else
   {
      m_mtx_pool.unlock();
      logmsg = new LogMessage(areaName, modname, caller, level, msg, code );
   }

   return logmsg;
}


void LogChannel::disposeMessage(LogMessage* msg)
{
   m_mtx_pool.lock();
   if( m_poolSize >= MAX_MSG_POOL )
   {
      m_mtx_pool.unlock();
      delete msg;
   }
   else {
      msg->m_next = m_pool;
      m_pool = msg;
      m_poolSize++;
      m_mtx_pool.unlock();
   }
}




}
}

/* end of logchannel.cpp */

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

namespace Falcon {

LogChannel::LogChannel( uint32 l ):
   m_refCount( 1 ),
   m_msg_head(0),
   m_msg_tail(0),
   m_terminate(false),
   m_level( l )
   {
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


void LogChannel::log( LogArea* area, uint32 l, const String& msg )
{
   if ( l <= m_level )
   {
      // delegate formatting to the other thread.
      LogMessage* lmsg = new LogMessage( area->name(), l, msg );

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

      while( m_msg_head !=0 )
      {
         LogMessage* msg = m_msg_head;
         m_msg_head = m_msg_head->m_next;
         m_msg_mtx.unlock();

         //TODO: format the message
         writeLogEntry( msg->m_msg );
         delete msg;
         m_msg_mtx.lock();
      }

      // we emptied the queue.
      m_msg_tail =0;
      m_msg_mtx.unlock();
   }

   return 0;
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


void LogArea::log( uint32 level, const String& msg )
{
   m_mtx_chan.lock();
   ChannelCarrier* cc = m_head_chan;
   while( cc != 0 )
   {
      cc->m_channel->log( this, level, msg );
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

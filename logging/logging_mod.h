/*
   FALCON - The Falcon Programming Language.
   FILE: logging_mod.h

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_logging_mod_H
#define flc_logging_mod_H

#include <falcon/setup.h>
#include <falcon/mt.h>
#include <falcon/string.h>

namespace Falcon
{

#define LOGLEVEL_FATAL 0
#define LOGLEVEL_ERROR 1
#define LOGLEVEL_WARN  2
#define LOGLEVEL_INFO  3
#define LOGLEVEL_LOW   4
#define LOGLEVEL_DEBUG 5
#define LOGLEVEL_D1    6
#define LOGLEVEL_D2    7

#define LOGLEVEL_ALL   100

class LogArea;
class Stream;

/** Abstract base class for logging channels. */
class FALCON_DYN_CLASS LogChannel: public Runnable
{
   volatile int m_refCount;
   friend class LogArea;

   Mutex m_msg_mtx;
   Event m_message_incoming;
   SysThread* m_thread;

   class LogMessage
   {
   public:
      String m_areaName;
      int m_level;
      String m_msg;
      LogMessage* m_next;

      LogMessage( const String& areaName, int level, const String& msg ):
         m_areaName(msg),
         m_level( level ),
         m_msg( msg ),
         m_next(0)
         {}
   };

   LogMessage* m_msg_head;
   LogMessage* m_msg_tail;
   bool m_terminate;

   void start();
   void stop();



protected:
   uint32 m_level;
   String m_format;

   /** Override this to send a pre-formatted message to the output device */
   virtual void writeLogEntry( const String& entry ) = 0;
   virtual ~LogChannel();
public:

   LogChannel( uint32 l = LOGLEVEL_ALL ):
      m_refCount( 1 ),
      m_msg_head(0),
      m_msg_tail(0),
      m_terminate(false),
      m_level( l )
      {}

   LogChannel( const String &format, uint32 l = LOGLEVEL_ALL ):
      m_refCount( 1 ),
      m_msg_head(0),
      m_msg_tail(0),
      m_terminate(false),
      m_level( l ),
      m_format(format)
      {}

   void level( uint32 l ) { m_level = l; }
   uint32 level() const { return m_level; }

   void format( const String& fmt ) { m_format = fmt; }
   const String&  format() const { return m_format; }

   void incref();
   void decref();
   void log( LogArea* area, uint32 level, const String& msg );
   virtual void* run();
};

/** Area for logging.
 *
 */
class FALCON_DYN_CLASS LogArea
{
   volatile int m_refCount;
   String m_name;

   virtual ~LogArea();

   class ChannelCarrier
   {
   public:
      ChannelCarrier* m_next;
      ChannelCarrier* m_prev;

      LogChannel* m_channel;

      ChannelCarrier( LogChannel* chn ):
         m_channel( chn )
      {}
   };

   ChannelCarrier* m_head_chan;
   Mutex m_mtx_chan;

public:
   LogArea( const String& name ):
      m_refCount(1),
      m_name( name ),
      m_head_chan( 0 )
   {}

   void log( uint32 level, const String& msg );

   void incref();
   void decref();

   const String& name() const { return m_name; }

   void addChannel( LogChannel* chn );
   void removeChannel( LogChannel* chn );
};


class FALCON_DYN_CLASS LogChannelStream: public LogChannel
{
protected:
   Stream* m_stream;
   void writeLogEntry( const String& entry );

public:
   LogChannelStream( Stream* s, int level=LOGLEVEL_ALL ):
      LogChannel( level ),
      m_stream( s )
      {}

   LogChannelStream( Stream* s, const String &fmt, int level=LOGLEVEL_ALL ):
      LogChannel( fmt, level ),
      m_stream( s )
      {}

   virtual ~LogChannelStream();
};

}

#endif

/* end of logging_mod.h */

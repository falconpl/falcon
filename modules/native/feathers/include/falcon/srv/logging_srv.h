/*
   FALCON - The Falcon Programming Language.
   FILE: logging_srv.h

   Logging module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Sep 2009 16:42:41 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef flc_logging_srv_H
#define flc_logging_srv_H

#include <falcon/setup.h>
#include <falcon/service.h>
#include <falcon/string.h>
#include <falcon/timestamp.h>
#include <falcon/mt.h>

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
class FileStream;

/** Abstract base class for logging channels. */
class LogChannel: public Runnable
{
   volatile int m_refCount;
   friend class LogArea;

   Mutex m_msg_mtx;
   Event m_message_incoming;
   SysThread* m_thread;

protected:

   TimeStamp m_ts;
   numeric m_startedAt;

protected:

   class LogMessage
   {
   public:
      String m_areaName;
      String m_modName;
      String m_caller;
      int m_level;
      String m_msg;
      uint32 m_code;
      LogMessage* m_next;

      LogMessage( const String& areaName, const String& modname, const String& caller, int level, const String& msg, uint32 code = 0 ):
         m_areaName( areaName ),
         m_modName( modname ),
         m_caller( caller ),
         m_level( level ),
         m_msg( msg ),
         m_code( code ),
         m_next(0)
         {}
   };

   virtual void pushFront( LogMessage* lm );
   virtual void pushBack( LogMessage* lm );

private:
   LogMessage* m_msg_head;
   LogMessage* m_msg_tail;
   bool m_terminate;
   bool m_bTsReady;

   void updateTS()
   {
      if( ! m_bTsReady )
      {
         m_bTsReady = true;
         m_ts.currentTime();
      }
   }

   void start();
   bool expandMessage( LogMessage* msg, const String& fmt, String& target );

protected:
   uint32 m_level;
   String m_format;

   virtual void stop();
   /** Override this to send a pre-formatted message to the output device */
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg ) = 0;
   virtual ~LogChannel();
public:

   LogChannel( uint32 l = LOGLEVEL_ALL );
   LogChannel( const String &format, uint32 l = LOGLEVEL_ALL );

   inline void level( uint32 l ) { m_level = l; }
   inline uint32 level() const { return m_level; }

   virtual void setFormat( const String& fmt );
   virtual void getFormat( String& fmt );

   virtual void incref();
   virtual void decref();

   virtual void log( LogArea* area, uint32 level, const String& msg, uint32 code = 0 );

   inline void log( uint32 level, const String& msg, uint32 code = 0 ) { log( "", "", level, msg, code ); }
   inline void log( const String& tgt, const String& source, uint32 level, const String& msg, uint32 code = 0 )
   {
      log( tgt, source, "", level, msg );
   }
   virtual void log( const String& tgt, const String& source, const String& function, uint32 level, const String& msg, uint32 code = 0 );
   virtual void* run();

};

/** Area for logging.
 *
 */
class LogArea: public BaseAlloc
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
   mutable Mutex m_mtx_chan;

public:
   LogArea( const String& name ):
      m_refCount(1),
      m_name( name ),
      m_head_chan( 0 )
   {}

   virtual void log( uint32 level, const String& msg, uint32 code = 0 ) const
   {
      log( level, "", "", msg, code );
   }

   virtual void log( uint32 level, const String& source, const String& msg, uint32 code = 0 ) const
   {
      log( level, source, "", msg, code );
   }

   virtual void log( uint32 level, const String& source, const String& func, const String& msg, uint32 code = 0 ) const;

   virtual void incref();
   virtual void decref();

   virtual const String& name() const { return m_name; }

   virtual void addChannel( LogChannel* chn );
   virtual void removeChannel( LogChannel* chn );
   virtual int minlog() const;
};


class LogChannelStream: public LogChannel
{
protected:
   Stream* m_stream;
   bool m_bFlushAll;
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual ~LogChannelStream();

public:
   LogChannelStream( Stream* s, int level=LOGLEVEL_ALL );
   LogChannelStream( Stream* s, const String &fmt, int level=LOGLEVEL_ALL );


   inline bool flushAll() const { return m_bFlushAll; }
   inline void flushAll( bool b ) { m_bFlushAll = b; }
};


class LogChannelFiles: public LogChannel
{
private:
   void inner_rotate();
   TimeStamp m_opendate;


protected:
   FileStream* m_stream;
   bool m_bFlushAll;

   String m_path;
   int64 m_maxSize;
   int32 m_maxCount;
   bool m_bOverwrite;
   int32 m_maxDays;
   bool m_isOpen;


   virtual void expandPath( int32 number, String& path );
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual ~LogChannelFiles();

public:
   LogChannelFiles( const String& path, int level=LOGLEVEL_ALL );
   LogChannelFiles( const String& path, const String &fmt, int level=LOGLEVEL_ALL );

   /** Overloads the base log request opening the channel if necessary */
   virtual void log( const String& tgt, const String& source, const String& function, uint32 level, const String& msg, uint32 code = 0 );

   /** Opens the log. May throw an IoError. */
   virtual void open();

   /** Truncates the log. May throw an IoError. */
   virtual void reset();

   /** Perform a rollover. */
   virtual void rotate();

   inline LogChannelFiles& flushAll( bool b ) { m_bFlushAll = b; return *this;}
   inline LogChannelFiles& maxSize( int64 ms ) { m_maxSize = ms; return *this;}
   inline LogChannelFiles& maxCount( int32 mc ) { m_maxCount = mc; return *this;}
   inline LogChannelFiles& overwrite( bool ow ) { m_bOverwrite = ow; return *this;}
   inline LogChannelFiles& maxDays( int32 md ) { m_maxDays = md; return *this;}

   inline bool flushAll() const { return m_bFlushAll; }
   inline int64 maxSize() const { return m_maxSize; }
   inline int32 maxCount() const { return m_maxCount;}
   inline bool overwrite() const { return m_bOverwrite;}
   inline int32 maxDays() const { return m_maxDays;}
   inline const String& path() const { return m_path;}
};


/** Logging for Syslog (POSIX) or Event Logger (MS-Windows).  */
class LogChannelSyslog: public LogChannel
{
private:
   void* m_sysdata;

protected:
   String m_identity;
   uint32 m_facility;

   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg );
   virtual void init();

   virtual ~LogChannelSyslog();

public:
   LogChannelSyslog( const String& identity, uint32 facility = 0, int level=LOGLEVEL_ALL );
   LogChannelSyslog( const String& identity, const String &fmt, uint32 facility = 0, int level=LOGLEVEL_ALL );
};


#define LOGSERVICE_NAME "LogService"

/** Publishes the Log system as "LogService".
 *
 */
class LogService: public Service
{
public:
   LogService();

   virtual LogArea* makeLogArea( const String& name ) const;
   virtual LogChannelStream* makeChnStream( Stream* s, int level=LOGLEVEL_ALL ) const;
   virtual LogChannelStream* makeChnStream( Stream* s, const String &fmt, int level=LOGLEVEL_ALL ) const;

   virtual LogChannelSyslog* makeChnSyslog( const String& identity, uint32 facility = 0, int level=LOGLEVEL_ALL ) const;
   virtual LogChannelSyslog* makeChnSyslog( const String& identity, const String &fmt, uint32 facility = 0, int level=LOGLEVEL_ALL ) const;

   virtual LogChannelFiles* makeChnlFiles( const String& path, int level=LOGLEVEL_ALL ) const;
   virtual LogChannelFiles* makeChnFiles( const String& path, const String &fmt, int level=LOGLEVEL_ALL ) const;

};

}

#endif

/* end of logging_srv.h */

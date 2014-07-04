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

#ifndef FALCON_FEATHERS_LOGCHANNEL_H
#define FALCON_FEATHERS_LOGCHANNEL_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/mt.h>
#include <falcon/refcounter.h>
#include <falcon/timestamp.h>

namespace Falcon {
namespace Feathers {

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

/** Abstract base class for logging channels. */
class LogChannel: public Runnable
{
public:

   inline void level( uint32 l ) { m_level = l; }
   inline uint32 level() const { return m_level; }

   virtual void setFormat( const String& fmt );
   virtual void getFormat( String& fmt );

   virtual void log( LogArea* area, uint32 level, const String& msg, uint32 code = 0 );

   inline void log( uint32 level, const String& msg, uint32 code = 0 ) { log( "", "", level, msg, code ); }
   inline void log( const String& tgt, const String& source, uint32 level, const String& msg, uint32 code = 0 )
   {
      log( tgt, source, "", level, msg, code );
   }
   virtual void log( const String& tgt, const String& source, const String& function, uint32 level, const String& msg, uint32 code = 0 );
   virtual void* run();

   /** Closes the channel.
    *\return false if the method was already called (atomic check).
    *
    * This method perform a phisical closing of the underlying service.
    *
    * The base version of the method also ensures that received messages are
    * discarded, atomically. A subclass invoking the base class is sure that
    * its writeLogEntry() method won't be called after the base method is invoked.
    *
    * However, the main thread still keeps running, and must be stopped separately with
    * the stop() method. Stop doesn't call close(); the proper close() method, if necessary
    * must be called by the destructor of the subclass, if it wasn't previously called.
    *
    */
   virtual bool close();
   bool closed() const;

   uint32 currentMark() const { return m_mark; }
   void gcMark( uint32 m ) { m_mark = m; }

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

   TimeStamp m_ts;
   numeric m_startedAt;
   uint32 m_level;
   String m_format;

   /** Override this to send a pre-formatted message to the output device */
   virtual void writeLogEntry( const String& entry, LogMessage* pOrigMsg ) = 0;

   LogChannel( uint32 l = LOGLEVEL_ALL );
   LogChannel( const String &format, uint32 l = LOGLEVEL_ALL );

   void pushFront( LogMessage* lm );
   void pushBack( LogMessage* lm );

   void start();
   void stop();

   virtual ~LogChannel();

private:
   static const int MAX_MSG_POOL=32;

   mutable Mutex m_msg_mtx;
   Event m_message_incoming;
   SysThread* m_thread;

   LogMessage* m_msg_head;
   LogMessage* m_msg_tail;

   mutable Mutex m_mtx_pool;
   LogMessage* m_pool;
   int32 m_poolSize;

   bool m_terminate;
   bool m_bTsReady;
   bool m_bClosed;

   uint32 m_mark;

   void updateTS()
   {
      if( ! m_bTsReady )
      {
         m_bTsReady = true;
         m_ts.currentTime();
      }
   }

   bool expandMessage( LogMessage* msg, const String& fmt, String& target );

   LogMessage* allocMessage(const String& areaName, const String& modname, const String& caller, int level, const String& msg, uint32 code = 0 );
   void disposeMessage(LogMessage* msg);

private:
   FALCON_REFERENCECOUNT_DECLARE_INCDEC_NOEXPORT(LogChannel);
};

}
}

#endif

/* end of logchannel.h */

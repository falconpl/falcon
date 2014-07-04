/*
   FALCON - The Falcon Programming Language.
   FILE: logchannel_files.h

   Logging module -- log channel interface (for self-rolling files)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_LOGCHANNEL_FILES_H
#define FALCON_FEATHERS_LOGCHANNEL_FILES_H

#include "logchannel.h"
#include <falcon/textwriter.h>
#include <falcon/timestamp.h>
#include <falcon/mt.h>

namespace Falcon {
namespace Feathers {

/** Logging channel using self-rotating files
*
* By default, the text encoding using for logging is
* utf-8. You can change the default by invoking the encoding()
* method prior assigning the channel to a log area.
*/

class LogChannelFiles: public LogChannel
{

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

   virtual bool close();

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

   /** Gets the text encoding used on the log files..
    *
    */
   String encoding() const;

   /** Sets the text encoding used on the log files..
    * @return true if the encoding is a known encoding name, false otherwise.
    */
   bool encoding( const String& enc );

protected:
   TextWriter* m_stream;
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

private:
   mutable Mutex m_mtx_open;
   TimeStamp m_opendate;
   String m_encoding;

   void inner_rotate();
};

}
}

#endif

/* end of logchannel_files.h */


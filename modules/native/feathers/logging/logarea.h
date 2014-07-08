/*
   FALCON - The Falcon Programming Language.
   FILE: logarea.h

   Logging module -- log area interface
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 23 Aug 2013 04:43:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_LOGAREA_H
#define FALCON_FEATHERS_LOGAREA_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/refcounter.h>

namespace Falcon {
namespace Feathers {

class LogChannel;

/** Area for logging.
 *
 */
class LogArea
{
public:
   LogArea( const String& name );

   virtual void log( uint32 level, const String& msg, uint32 code = 0 ) const
   {
      log( level, "", "", msg, code );
   }

   virtual void log( uint32 level, const String& source, const String& msg, uint32 code = 0 ) const
   {
      log( level, source, "", msg, code );
   }

   virtual void log( uint32 level, const String& source, const String& func, const String& msg, uint32 code = 0 ) const;

   virtual const String& name() const { return m_name; }
   virtual void name( const String& n ){ m_name = n; }

   virtual void addChannel( LogChannel* chn );
   virtual void removeChannel( LogChannel* chn );
   virtual int minlog() const;

   void gcMark( uint32 m ) { m_mark = m; }
   uint32 currentMark() const { return m_mark; }

private:
   String m_name;
   uint32 m_mark;

   virtual ~LogArea();
   class Private;
   Private* _p;

   FALCON_REFERENCECOUNT_DECLARE_INCDEC_NOEXPORT(LogArea);
};

}
}

#endif

/* end of logarea.h */


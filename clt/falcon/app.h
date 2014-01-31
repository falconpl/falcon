/*
   FALCON - The Falcon Programming Language.
   FILE: falcon.h

   Falcon compiler and interpreter
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 23 Mar 2009 18:57:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Main Falcon.
*/

#ifndef FALCON_CLT_H
#define FALCON_CLT_H

#include <falcon/falcon.h>

#include "options.h"

namespace Falcon {

   class Log;

class FalconApp: public Application
{

public:
   int m_exitValue;
   FalconOptions m_options;

   /**
    * The logger for the Falcon application.
    *
    * \TODO Add stream/stderr control in command line.
    */
   class Logger: public Log::Listener
   {
   public:
      Logger();
      virtual ~Logger();
      TextWriter* m_logfile;

   protected:

      virtual void onMessage( int fac, int lvl, const String& message );

   };

   Logger* m_logger;

   FalconApp();
   ~FalconApp();
   
   void guardAndGo( int argc, char* argv[] );
   void interactive();
   void testMode();
   void launch( const String& script, int argc, char* argv[], int pos );
   void evaluate( const String& string, bool stdin );

   void configureVM( VMachine& vm, Process* prc, Log* log = 0 );

private:

   Debugger m_dbg;
};

} // namespace Falcon

#endif

/* end of falcon.h */

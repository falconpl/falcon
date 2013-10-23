/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd.h

   Micro HTTPD server providing Falcon scripts on the web.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 23 Feb 2010 22:09:02 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "falhttpd.h"
#include <falcon/stream.h>
#include <falcon/textwriter.h>

namespace Falcon {

FalhttpdApp::LogListener::LogListener( Stream* log ):
         m_logfile(0)
{
   m_logfile = new TextWriter( log );
}

FalconApp::Logger::~Logger()
{
   delete m_logfile;
}

void FalconApp::Logger::onMessage( int fac, int lvl, const String& message )
{
   String tgt;
   Log::addTS(tgt);
   tgt += " ";
   tgt += message;
   tgt += "\n";

   m_logfile->writeLine( tgt );
   m_logfile->flush();
}

}
/* end of falhttpd_loglistener.cpp */

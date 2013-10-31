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

FalhttpdApp::LogListener::~LogListener()
{
   delete m_logfile;
}

void FalhttpdApp::LogListener::onMessage( int , int lvl, const String& message )
{
   String tgt;
   Log::addTS(tgt);
   tgt += " ";
   tgt += Log::levelToString(lvl);
   tgt += " ";
   tgt += message;

   m_logfile->writeLine( tgt );
   m_logfile->flush();
}

}
/* end of falhttpd_loglistener.cpp */

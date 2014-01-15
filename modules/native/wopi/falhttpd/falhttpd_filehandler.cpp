/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_filehandler.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Handler for requests of files.
   This also determines the MIME type of served files.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:16:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "falhttpd_rh.h"
#include "falhttpd_filehandler.h"
#include "falhttpd_client.h"

#include <falcon/timestamp.h>

namespace Falcon {

FileHandler::FileHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
}


FileHandler::~FileHandler()
{
}


void FileHandler::serve()
{
   String sMimeType;
   Falcon::WOPI::Request* req  = m_client->request();

   // read the rest of the request.
   try
   {
      req->parse( m_client->stream() );
   }
   catch(Error* error)
   {
      m_client->replyError( 400, error->describe(true) );
      error->decref();
      return;
   }

   // Get the document -- for now a very simple thing
   if ( ! m_client->options().findMimeType( m_sFile, sMimeType ) )
   {
      LOGI("Unknown mime type for file " + m_sFile );
      sMimeType = "text/plain; charset=" + m_client->options().m_sTextEncoding;
   }
   else
   {
      LOGD("Mapping MIME type " + m_sFile + " -> " + sMimeType );
   }

   // Send the file
   FileStat stats;
   if( ! Engine::instance()->vfs().readStats( m_sFile, stats, true ) )
   {
      m_client->replyError( 403 );
      return;
   }

   try
   {
      Stream* fs = Engine::instance()->vfs().openRO(m_sFile);
      LOGI( "Sending file "+ m_sFile );

      // ok we can serve the file
      WOPI::Reply* rep = m_client->reply();


      TimeStamp now;
      now.currentTime();
      rep->setContentType(sMimeType);
      rep->setHeader("Date", now.toRFC2822() );
      TimeStamp ts(Date(stats.mtime(),0));
      rep->setHeader( "Last-Modified", ts.toRFC2822() );

      char buffer[4096];
      int len = fs->read( buffer, 4096 );
      while( len > 0 )
      {
        m_client->sendData( buffer, len );
        len = fs->read( buffer, 4096 );
      }

      if ( len < 0 )
      {
        // error!
        LOGW( "Error while reading file "+ m_sFile );
        m_client->replyError( 403 );
      }

   }
   catch( Error* e )
   {
     LOGW( "Can't open file "+ m_sFile );
     m_client->replyError( 403 );
   }
}

}

/* end of falhttpd_filehandler.cpp */

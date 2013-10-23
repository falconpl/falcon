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


void FileHandler::serve( Falcon::WOPI::Request* )
{
   String sMimeType;

   // Get the document -- for now a very simple thing
   if ( ! m_client->options().findMimeType( m_sFile, sMimeType ) )
   {
      sMimeType = "unknown";
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
      String sReply = "HTTP/1.0 200 OK\r\n";

      TimeStamp now;
      now.currentTime();
      sReply += "Content-Type: " + sMimeType + "; charset=" + m_client->options().m_sTextEncoding + "\r\n";
      sReply += "Date: " + now.toRFC2822() + "\r\n";
      TimeStamp ts(Date(stats.mtime(),0));
      sReply += "Last-Modified: " + ts.toRFC2822() + "\r\n";

      sReply += "\r\n";
      // content length not strictly necessary now

      m_client->sendData( sReply );
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
      return;
   }

}

}

/* end of falhttpd_filehandler.cpp */

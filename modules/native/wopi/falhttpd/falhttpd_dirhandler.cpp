/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_dirhandler.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Request handler processing access to directories.
   Mimetype is text/html; charset=utf-8

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Mar 2010 12:16:03 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "falhttpd.h"
#include "falhttpd_dirhandler.h"
#include "falhttpd_client.h"

#include <falcon/wopi/stream_ch.h>
#include <falcon/wopi/reply.h>

#include <falcon/sys.h>

namespace Falcon {

DirHandler::DirHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
}


DirHandler::~DirHandler()
{
}


void DirHandler::serve()
{
   // open the directory
   Falcon::Directory *de = 0;
   WOPI::Request* req  = m_client->request();

   // read the rest of the request.
   try
   {
      req->parse( m_client->stream() );
   }
   catch( Error* error )
   {
      m_client->errhand().replyError( m_client, 400, error->describe(true) );
      error->decref();
      return;
   }

   try {
      de = Engine::instance()->vfs().openDir( m_sFile );
      Falcon::String thisDir = m_sFile.subString( m_client->options().m_homedir.length() );

      if( ! thisDir.endsWith("/") )
      {
         thisDir +="/";
      }

      Falcon::String title =
            "<html><head><title>Index of "+ thisDir + "</title></head>"
            "<body>\n<h1>Index of "+ thisDir + "</h1>\n<p>";

      m_client->sendData( title );

      Falcon::String fname;
      while ( de->read(fname) )
      {
         if ( fname == "." )
            continue;

         Falcon::String loc;
         if( fname == ".." )
         {
            if( thisDir == "/" )
               continue;
         }

         Falcon::String entry = "<br/><a href=\""+ thisDir + fname +"\">" + fname + "</a>\n";
         m_client->sendData( entry );
      }

      m_client->sendData( "\n</body>\n</html>" );
   }
   catch(Error* e )
   {
      LOGW( "Can't open directory " + m_sFile );
      m_client->errhand().replyError( m_client, 403, "" );
   }

   delete de;
}

}

/* end of falhttpd_dirhandler.cpp */

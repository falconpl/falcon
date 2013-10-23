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

#include <falcon/sys.h>

namespace Falcon {

DirHandler::DirHandler( const Falcon::String& sFile, FalhttpdClient* cli ):
      FalhttpdRequestHandler( sFile, cli )
{
}


DirHandler::~DirHandler()
{
}


void DirHandler::serve( WOPI::Request* )
{
   // open the directory
   Falcon::Directory *de = 0;
   try {
      de = Engine::instance()->vfs().openDir( m_sFile );
      Falcon::String thisDir = m_sFile.subString( m_client->options().m_homedir.length() );

      if( thisDir.endsWith("/") && thisDir.length() >  1 )
      {
         thisDir.remove(thisDir.length()-1, 1);
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

            Falcon::uint32 pos = thisDir.rfind("/");
            if ( pos == Falcon::String::npos )
               loc = "";
            else
               loc = thisDir.subString(0,pos);
         }
         else
         {
            loc = thisDir;
         }

         Falcon::String entry = "<br/><a href=\""+ loc + "/" + fname +"\">" + fname + "</a>\n";
         m_client->sendData( entry );
      }

      m_client->sendData( "\n</body>\n</html>" );
   }
   catch(Error* e )
   {
      LOGW( "Can't open directory " + m_sFile );
      m_client->replyError( 403 );
   }

   delete de;
}

}

/* end of falhttpd_dirhandler.cpp */

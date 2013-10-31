/*
   FALCON - The Falcon Programming Language.
   FILE: stream_ch.cpp

   Falcon CGI program driver - cgi-based reply specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/stream_ch.h>
#include <falcon/stream.h>
#include <falcon/autocstring.h>

namespace Falcon {
namespace WOPI {

StreamCommitHandler::StreamCommitHandler( Stream* stream ):
         m_stream(stream)
{
   stream->incref();
}

StreamCommitHandler::~StreamCommitHandler()
{
   m_stream->decref();
}


void StreamCommitHandler::startCommit( Reply* rep )
{
   Falcon::String sRep;
   sRep.A("HTTP/1.1 ").N( rep->status() ).A(" ").A( rep->reason() ).A("\r\n");
   m_stream->write( sRep.c_ize(), sRep.size() );
}

void StreamCommitHandler::commitHeader( Reply*, const Falcon::String& name, const Falcon::String& value )
{
   AutoCString header(name + ": " + value + "\r\n");
   m_stream->write( header.c_str(), header.length() );
}

void StreamCommitHandler::endCommit( Reply* )
{
   m_stream->write( "\r\n", 2 );
   m_stream->flush();
}

}
}

/* end of cgi_commithandler.cpp */

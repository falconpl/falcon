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

StreamCommitHandler::StreamCommitHandler()
{
}

StreamCommitHandler::~StreamCommitHandler()
{
}


void StreamCommitHandler::startCommit( Reply* rep, Stream* stream )
{
   Falcon::String sRep;
   sRep.A("HTTP/1.1 ").N( rep->status() ).A(" ").A( rep->reason() );
   m_output->writeString( sRep + "\r\n" );
}

void StreamCommitHandler::commitHeader( Reply*, Stream* stream, const Falcon::String& name, const Falcon::String& value )
{
   AutoCString header(name + ": " + value + "\r\n");
   stream->write( header.c_str(), header.length() );
}

void StreamCommitHandler::endCommit( Reply*, Stream* stream )
{
   stream->write( "\r\n", 2 );
   stream->flush();
}

}
}

/* end of cgi_commithandler.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: falcgi_make_streams.cpp

   Falcon CGI program driver - CGI specific stream provider.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 17:35:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <cgi_make_streams.h>
#include <falcon/stdstreams.h>
#include <falcon/fstream.h>

Falcon::Stream* makeOutputStream()
{
   return new Falcon::StdOutStream;
}

Falcon::Stream* makeInputStream()
{
   return new Falcon::StdInStream;
}

Falcon::Stream* makeErrorStream()
{
   return Falcon::stdErrorStream();
}

/* end of falcgi_make_streams.cpp */

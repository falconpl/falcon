/*
   FALCON - The Falcon Programming Language.
   FILE: falhttpd_ch.cpp

   Micro HTTPD server providing Falcon scripts on the web.

   Commit handler -- falhttpd specific reply commit rules.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Mar 2010 21:31:37 +0100
s
   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include "falhttpd_ch.h"

namespace Falcon {


Falhttpd_CommitHandler::Falhttpd_CommitHandler()
{
}

Falhttpd_CommitHandler::~Falhttpd_CommitHandler()
{
}

void Falhttpd_CommitHandler::startCommit( WOPI::Reply* reply, Stream* )
{
   m_headers.A("HTTP/1.1 ").N( reply->status() ).A( reply->reason() ).A("\r\n");
}

void Falhttpd_CommitHandler::commitHeader( WOPI::Reply*, Stream*, const String& hname, const String& hvalue )
{
   m_headers += hname + ": " + hvalue + "\r\n";
}

void Falhttpd_CommitHandler::endCommit( WOPI::Reply*, Stream* tgt )
{
   m_headers += "\r\n";
   length_t written = m_headers.size();

   // overkill, but...
   while( written < m_headers.size() )
   {
      written += tgt->write( m_headers.getRawStorage() + written, m_headers.size() - written  );
   }
}


}

/* falhttpd_ch.cpp */

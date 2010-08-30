/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_reply.cpp

   Falcon CGI program driver - cgi-based reply specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <cgi_reply.h>
#include <cgi_make_streams.h>

CGIReply::CGIReply( const Falcon::CoreClass* cls ):
   Reply( cls ),
   m_output( 0 )
{
}

CGIReply::~CGIReply()
{
}

void CGIReply::init( )
{
}


void CGIReply::startCommit()
{
   Falcon::String sRep;
   /*sRep.A("HTTP/1.1 ").N( m_nStatus ).A(" ").A(m_sReason);
   m_output->writeString( sRep + "\r\n" );*/

   m_bHeadersSent = true;
}

Falcon::Stream* CGIReply::makeOutputStream()
{
   m_output = ::makeOutputStream();
   return m_output;
}

void CGIReply::commitHeader( const Falcon::String& name, const Falcon::String& value )
{
   m_output->writeString( name + ": " + value + "\r\n" );
}

void CGIReply::endCommit()
{
   m_output->writeString( "\r\n" );
   m_output->flush();
}

Falcon::CoreObject* CGIReply::factory( const Falcon::CoreClass* cls, void* , bool )
{
   return new CGIReply( cls );
}


/* end of cgi_reply.cpp */

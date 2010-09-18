/*
   FALCON - The Falcon Programming Language.
   FILE: apache_reply.cpp

   Web Oriented Programming Interface

   Apache-specific reply object.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 20 Feb 2010 10:43:35 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/autocstring.h>
#include <apr_strings.h>

#include "apache_reply.h"
#include "apache_output.h"
#include "apache_stream.h"

ApacheReply::ApacheReply( const Falcon::CoreClass* base ):
   Reply( base ),
   m_request( 0 ),
   m_aout(0)
{
}


ApacheReply::~ApacheReply()
{
}

void ApacheReply::init( request_rec *r, ApacheOutput* output )
{
   m_aout = output;
   m_request = r;
}

void ApacheReply::startCommit()
{
   m_request->status = m_nStatus;
   Falcon::AutoCString csline( m_sReason );
   m_request->status_line = apr_pstrdup ( m_request->pool, csline.c_str() );
}


Falcon::Stream* ApacheReply::makeOutputStream()
{
   return new ApacheStream( m_aout );
}


void ApacheReply::endCommit()
{
   // nothing to do
}

void ApacheReply::commitHeader( const Falcon::String& name, const Falcon::String& value )
{
   Falcon::AutoCString cvalue( value );

   if ( name.compareIgnoreCase( "Content-Type" ) == 0 )
   {
      m_request->content_type = apr_pstrdup ( m_request->pool, cvalue.c_str() );
   }
   else
   {
      Falcon::AutoCString cname( name );
      apr_table_add( m_request->headers_out, cname.c_str(), cvalue.c_str() );
   }
}

Falcon::CoreObject* ApacheReply::factory( const Falcon::CoreClass* cls, void* , bool  )
{
   return new ApacheReply( cls );
}


/* end of apache_reply.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: cgi_request.cpp

   Falcon CGI program driver - cgi-based request specialization.

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 21 Feb 2010 12:19:38 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "cgi_request.h"
#include "cgi_reply.h"

#include <falcon/wopi/file_sm.h>

#include <falcon/sys.h>
#include <falcon/fstream.h>
#include <falcon/coredict.h>
#include <falcon/stdstreams.h>
#include <falcon/vm.h>
#include <falcon/livemodule.h>

#include <cctype>

CGIRequest::CGIRequest( const Falcon::CoreClass* cls ):
   CoreRequest( cls ),
   m_cration_time(0),
   m_post_length(0),
   m_vmOwner(0)
{
   m_provider = "CGI";
}

CGIRequest::~CGIRequest()
{
   // Actually, it does nothing if sessions have already been released.
   if( m_sm != 0 && m_base != 0 )
   {
      m_sm->releaseSessions( m_base->sessionToken() );
   }
}


void CGIRequest::init( Falcon::Stream* input, Falcon::CoreClass* upld_cls, Falcon::WOPI::Reply* r, Falcon::WOPI::SessionManager* sm )
{
   CoreRequest::init( upld_cls, r, sm );

   // First; suck all the environment variables that we need.
   Falcon::Sys::_enumerateEnvironment( &handleEnvStr, this );

   // a bit of post-processing
   if ( m_base->parsedUri().port() == "443" || m_base->parsedUri().port() == "https" )
   {
      m_base->parsedUri().scheme("https");
   }
   else
   {
      m_base->parsedUri().scheme("http");
   }

   m_base->m_request_time = 0;
   m_base->m_bytes_sent = 0;

   if ( m_base->m_method == "POST" )
   {
      // on error, proper fields are set.
      m_base->parseBody( input );
      processMultiPartBody();
   }
}

void CGIRequest::handleEnvStr( const Falcon::String& key, const Falcon::String& value, void *data )
{
   // First; suck all the environment variables that we need.
   CGIRequest* self = (CGIRequest*) data;
   Falcon::WOPI::Request* r = self->base();

   // Is this an header transformed in an env-var?
   if( key.startsWith("HTTP_") )
   {
      self->addHeaderFromEnv( key, value );
   }
   else if( key == "AUTH_TYPE" )
   {
      r->m_ap_auth_type = value;
   }
   else if( key == "CONTENT_TYPE" )
   {
      self->m_post_type = r->m_content_type = value;
      r->m_MainPart.addHeader( "Content-Type", value );
   }
   else if( key == "CONTENT_LENGTH" )
   {
      Falcon::int64 tgt;
      value.parseInt(tgt);
      r->m_content_length = (int) tgt;
      self->m_post_length = (int) tgt;
   }
   else if( key == "DOCUMENT_ROOT" )
   {
      // ....
   }
   else if( key == "GATEWAY_INTERFACE" )
   {
      // ....
   }
   else if( key == "PATH_INFO" )
   {
      r->m_path_info = value;
   }
   else if ( key == "QUERY_STRING" )
   {
      // it's part of the REQUEST_URI
   }
   else if( key == "REMOTE_ADDR" )
   {
      r->m_remote_ip = value;
   }
   else if( key == "REMOTE_PORT" )
   {
      //... not implemented
   }
   else if( key == "REMOTE_USER" )
   {
      r->m_user = value;
   }
   else if( key == "REQUEST_METHOD" )
   {
      r->m_method = value;
   }
   else if( key == "REQUEST_URI" )
   {
      r->setURI( value );
   }
   else if( key == "SCRIPT_FILENAME" )
   {
      r->m_filename = value;
   }
   else if( key == "SCRIPT_NAME" )
   {
      r->parsedUri().path( value );
   }
   else if( key == "SERVER_ADDR" )
   {
      // ....
   }
   else if( key == "SERVER_ADMIN" )
   {
      // ....
   }
   else if( key == "SERVER_NAME" )
   {
      r->parsedUri().host( value );
   }
   else if( key == "SERVER_PORT" )
   {
      r->parsedUri().port( value );
   }
   else if( key == "SERVER_PROTOCOL" )
   {
      r->m_protocol = value;
   }
   else if( key == "SERVER_SIGNATURE" )
   {
      // ....
   }
   else if( key == "SERVER_SOFTWARE" )
   {
      // ....
   }
}

void CGIRequest::addHeaderFromEnv( const Falcon::String& key, const Falcon::String& value )
{
   // discard "http_"
   Falcon::String sKey( key, 5 );

   // ... and transform the rest in "Content-Type" format
   bool bUpper = true;
   Falcon::uint32 len = sKey.length();

   for( Falcon::uint32 i = 0; i < len; ++i )
   {
      if( sKey[i] == '_' )
      {
         sKey.setCharAt(i, '-');
         bUpper = true;
      }
      else if ( bUpper )
      {
         sKey.setCharAt(i, toupper( sKey[i] ) );
         bUpper = false;
      }
      else
      {
         sKey.setCharAt(i, tolower( sKey[i] ) );
      }
   }

   // Ok, we can now add the thing to the dict
   m_base->headers()->put( new Falcon::CoreString( sKey ), new Falcon::CoreString( value ) );

   if ( key == "HTTP_COOKIE" )
   {
      Falcon::uint32 pos = 0;
      Falcon::uint32 pos1 = value.find(";");
      while( true )
      {
         Falcon::WOPI::Utils::parseQueryEntry( value.subString(pos,pos1), m_base->cookies()->items() );

         if( pos1 == Falcon::String::npos )
            break;

         pos = pos1+1;
         pos1 = value.find(";", pos );
      }

   }
   else if ( key == "HTTP_CONTENT_TYPE" )
   {
      m_base->m_content_type = value;
   }
   else if ( key == "HTTP_CONTENT_ENCODING" )
   {
      m_base->m_content_encoding = value;
   }
}

void CGIRequest::PostInitPrepare( Falcon::VMachine* vmowner )
{
   m_bPostInit = true;
   m_vmOwner = vmowner;
}

void CGIRequest::postInit()
{
   // ok, we can now pass to init.
   Falcon::Item* i_upld = m_vmOwner->findGlobalItem( "Uploaded" );
   fassert( i_upld != 0 );
   Falcon::Item* i_reply = m_vmOwner->findGlobalItem( "Reply" );
   fassert( i_reply != 0 );

   // configure using the main script
   Falcon::LiveModule* ms = m_vmOwner->mainModule();
   Falcon::WOPI::FileSessionManager* fsm = new Falcon::WOPI::FileSessionManager("");
   if( ms != 0 )
   {
      fsm->configFromModule( ms->module() );
      if( fsm->timeout() == 0 )
         fsm->timeout(600);
      fsm->startup();
   }

   CGIRequest::init( m_vmOwner->stdIn(), i_upld->asClass(),
         Falcon::dyncast<CGIReply*>(i_reply->asObject()),
         fsm );

   if( ms != 0 )
   {
      configFromModule( ms->module() );
   }

   m_base->startedAt( m_cration_time );
}

Falcon::CoreObject* CGIRequest::factory( const Falcon::CoreClass* cls, void* , bool )
{
   return new CGIRequest( cls );
}


/* end of cgi_request.cpp */

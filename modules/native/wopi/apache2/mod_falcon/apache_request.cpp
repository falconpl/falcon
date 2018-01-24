/*
   FALCON - The Falcon Programming Language.
   FILE: apache_request.cpp

   Falcon module for Apache 2
   Apache specific WOPI request
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 15:36:37 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/wopi/utils.h>
#include <falcon/string.h>
#include <falcon/streambuffer.h>

#include "mod_falcon.h"
#include "mod_falcon_config.h"
#include "apache_request.h"
#include "apache_stream.h"

#include <apr_strings.h>
#include <http_log.h>

//=====================================================================
// Helpers
//


static int table_to_dict(void *rec, const char *key, const char *value)
{
   struct ApacheRequest::table_callback *tbc = (struct ApacheRequest::table_callback *) rec;
   if ( apr_strnatcasecmp( key, "Cookie" ) == 0 )
   {
      // break the cookie.
      // we have a line.
      char *parse_value = apr_pstrdup( tbc->request->pool, value );
      char *last;
      char *token = apr_strtok(parse_value,";",&last);
      while ( token != NULL )
      {
         Falcon::String elem;
         elem.fromUTF8( token );
         Falcon::WOPI::Utils::parseQueryEntry( elem, *tbc->cookies );

         token = apr_strtok(NULL,";",&last);
      }
   }
   else
   {
      // anyhow, add the field.
      Falcon::CoreString* sKey = new Falcon::CoreString;
      Falcon::CoreString* sValue = new Falcon::CoreString;
      sKey->fromUTF8( key );
      sValue->fromUTF8( value );

      // we suppose the headers can't have []
      tbc->headers->put( sKey, sValue );

      // then check it
      if ( apr_strnatcasecmp( key, "Content-Type" ) == 0 )
      {
         // may be a multipart. -- if it is, boundary is mandatory.
         if ( strncmp( value, "multipart/form-data;", 20 ) == 0 )
         {
            // scan for the boundary key
            const char *keytok = value + 20;
            while ( *keytok == ' ' ) keytok++;

            if ( strncmp( keytok, "boundary", 8 ) == 0 )
            {
               keytok += 8;
               const char *base = keytok;
               while ( *keytok == ' ' || *keytok == '=' ) keytok++;

               // this ensures we have matched a "boundary =" and not i.e. a "boundary-nice"
               if ( keytok != base )
               {
                  // the real boundary has an extra "--" in front (as specified by mime)
                  // Also, it is preceded and followed by \r\n, but we don't precede it now,
                  // as we need a separate \r\n marker for line-based parts in multipart docs.
                  apr_cpystrn( tbc->boundary+2, keytok, BOUNDARY_SIZE-2 );
                  tbc->boundary[0] = '-';
                  tbc->boundary[1] = '-';
                  tbc->boundaryLen = strlen( tbc->boundary );
                  tbc->bIsMultiPart = true;
               }
            }
            // else ignore this field
         }
      }
      else if ( apr_strnatcasecmp( key, "Content-Length" ) == 0 )
      {
         tbc->contentLength = atoi( value );
      }
   }

   return 1;
}

//=====================================================================
// Main class
//

ApacheRequest::ApacheRequest( const Falcon::CoreClass* base ):
   CoreRequest( base )
{
   m_provider = "apache2";
}

void ApacheRequest::init( request_rec* request, Falcon::CoreClass* upd,
      Falcon::WOPI::Reply* reply,
      Falcon::WOPI::SessionManager* sm )
{
   CoreRequest::init( upd, reply, sm );
   m_request = request;
}

void ApacheRequest::process()
{
   //================================================
   // Parse headers
   struct table_callback tbc;
   tbc.headers = &m_base->headers()->items();
   tbc.cookies = &m_base->cookies()->items();
   tbc.bIsMultiPart = false;
   tbc.contentLength = 0;
   tbc.request = m_request;

   // parse all the headers
   apr_table_do( table_to_dict, &tbc, m_request->headers_in, NULL );

   //headers->put( new Falcon::CoreString(vm, "remaining"), request->remaining );

   //================================================
   // Parse get -- this is to be done anyhow.

   if ( m_request->args != 0 )
   {
      Falcon::WOPI::Utils::parseQuery( m_request->args, m_base->gets()->items() );
   }

   //================================================
   // Prepare the post parameters
   if ( m_request->method_number == M_POST )
   {
      m_base->m_content_length = tbc.contentLength;

      // but refuse to read the rest if content length is too wide
      if ( the_falcon_config->maxUpload > 0 &&
           the_falcon_config->maxUpload < tbc.contentLength )
      {
         Falcon::String reason = "Upload too large ";
         reason.writeNumber( (Falcon::int64) tbc.contentLength );
         reason += " (max ";
         reason.writeNumber( (Falcon::int64) the_falcon_config->maxUpload );
         reason += ")";

         m_base->posts()->put( new Falcon::CoreString(":error"), new Falcon::CoreString(reason) );
      }
      else
      {
         if( tbc.bIsMultiPart )
         {
            // apache already attaches the 2 extra  -- for the boundary begin.
            m_base->partHandler().setBoundary( Falcon::String( tbc.boundary+2, tbc.boundaryLen -2 ) );
         }

         // prepare for upload
         ApacheInputStream input( m_request );

         // on error, proper fields are set.
         m_base->parseBody( &input );
         processMultiPartBody();
      }
   }

   apr_uri_t* parsed_uri = &m_request->parsed_uri;
   if( parsed_uri->scheme ) m_base->parsedUri().scheme( parsed_uri->scheme );

   if( parsed_uri->user )
   {
      if( parsed_uri->password )
         m_base->parsedUri().userInfo( Falcon::String( parsed_uri->user ) + ":" + parsed_uri->password );
      else
         m_base->parsedUri().userInfo( parsed_uri->user );
   }

   if( parsed_uri->hostname ) m_base->parsedUri().host( parsed_uri->hostname );
   if( parsed_uri->port_str ) m_base->parsedUri().port( parsed_uri->port_str );
   if( parsed_uri->path )
   {
      m_base->parsedUri().path( parsed_uri->path );
      m_base->m_location.bufferize( parsed_uri->path );
   }
   if( parsed_uri->query ) m_base->parsedUri().query( parsed_uri->query );
   if( parsed_uri->fragment ) m_base->parsedUri().fragment( parsed_uri->fragment );

   // add in some other data elements from request_rec
   // avoiding redundant data
   if( m_request->method != 0 ) m_base->m_method = m_request->method;
   if( m_request->content_type != 0 ) m_base->m_content_type = m_request->content_type;
   if( m_request->content_encoding != 0 ) m_base->m_content_encoding = m_request->content_encoding;
   if( m_request->ap_auth_type != 0 ) m_base->m_ap_auth_type = m_request->ap_auth_type;
   if( m_request->user != 0 ) m_base->m_user = m_request->user;

   if( m_request->path_info != 0 ) m_base->m_path_info = m_request->path_info;
   if( m_request->filename != 0 ) m_base->m_filename = m_request->filename;
   if( m_request->protocol != 0 ) m_base->m_protocol = m_request->protocol;
   if( m_request->args != 0 ) m_base->m_args = m_request->args;
   if( m_request->connection->client_ip != 0 ) m_base->m_remote_ip = m_request->connection->client_ip;
   if( m_request->unparsed_uri != 0 ) m_base->m_sUri = m_request->unparsed_uri;

   m_base->m_request_time = (Falcon::int64) m_request->request_time;
   m_base->m_bytes_sent = (Falcon::int64) m_request->bytes_sent;
}

ApacheRequest::~ApacheRequest()
{
}

Falcon::CoreObject* ApacheRequest::factory( const Falcon::CoreClass* cls, void* , bool )
{
   return new ApacheRequest( cls );
}


/* end of apache_requeset.cpp */

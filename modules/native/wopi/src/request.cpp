/*
   FALCON - The Falcon Programming Language.
   FILE: request.cpp

   Web Oriented Programming Interface

   Object encapsulating requests.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 12:29:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/wopi/request.h>
#include <falcon/wopi/session_manager.h>
#include <falcon/wopi/utils.h>

#include <falcon/uri.h>
#include <falcon/error.h>
#include <falcon/engine.h>
#include <falcon/fstream.h>

// for memcpy
#include <string.h>
#include <stdio.h>


namespace Falcon {
namespace WOPI {


//==================================================================
// Request

Request::Request( ModuleWopi* host ):
   m_request_time( 0 ),
   m_bytes_sent( 0 ),
   m_content_length( -1 ),

   // protected
   m_gets( new ItemDict ),
   m_posts( new ItemDict ),
   m_cookies( new ItemDict ),
   m_headers( new ItemDict ),
   m_sSessionField( DEFAULT_SESSION_FIELD ),
   m_tempFiles( 0 ),
   m_startedAt(0.0)
{
   m_module = host;
   m_MainPart.setOwner( this );
}

Request::~Request()
{
}


bool Request::parse( Stream* input )
{
   if( ! parseHeader( input ) )
      return false;

   if( m_content_length > 0 )
      return parseBody( input );

   return true;
}

bool Request::parseHeader( Stream* input )
{
   if ( ! m_MainPart.parseHeader( input ) )
   {
      m_posts->insert(
               FALCON_GC_HANDLE(&(new String(":error"))->bufferize()),
               FALCON_GC_HANDLE(&(new String( m_MainPart.error() ))->bufferize())
          );
      return false;
   }

   // get the content type and encoding
   PartHandler::HeaderMap::const_iterator ci = m_MainPart.headers().find( "Content-Type" );
   if( ci != m_MainPart.headers().end() )
   {
      m_content_type = ci->second.rawValue();

      // default the content-length to 0 in case of non multipart data.
      if( m_content_type.find( "multipart/") != 0 )
         m_content_length = 0;

   }

   ci = m_MainPart.headers().find( "Content-Encoding" );
   if( ci != m_MainPart.headers().end() )
   {
      m_content_encoding = ci->second.rawValue();
   }

   ci = m_MainPart.headers().find( "Content-Length" );
   if( ci != m_MainPart.headers().end() )
   {
      ci->second.rawValue().parseInt( m_content_length );
      m_MainPart.setBodySize( m_content_length );
   }

   // parse the cookies.
   ci = m_MainPart.headers().find( "Cookie" );
   if( ci != m_MainPart.headers().end() )
   {
      HeaderValue::ParamMap::const_iterator pi = ci->second.parameters().begin();
      while( pi != ci->second.parameters().end() )
      {
         m_cookies->insert(
                        FALCON_GC_HANDLE(&(new String(pi->first))->bufferize()),
                        FALCON_GC_HANDLE(&(new String(pi->second))->bufferize())
                   );
         ++pi;
      }
   }

   ci = m_MainPart.headers().begin();
   while( ci != m_MainPart.headers().end() )
   {
      m_headers->insert(
              FALCON_GC_HANDLE(&(new String(ci->first))->bufferize()),
              FALCON_GC_HANDLE(&(new String(ci->second.rawValue()))->bufferize())
         );
      ++ci;
   }

   return true;
}


bool Request::parseBody( Stream* input )
{
   // prepare the POST data receive area
   m_MainPart.startMemoryUpload();
   //fprintf( stderr, "Content length: %d / %d\n", (int) m_content_length, (int) m_nMaxMemUpload );

   // Inform the part if it can use memory uploads for their subparts.
   if ( m_content_length != -1 &&
         (m_nMaxMemUpload > 0 && m_content_length < m_nMaxMemUpload) )
   {
      // This tell the children of the main part NOT TO create a temporary file
      // when they receive a file upload (the default).
      // Standard form fields are still received in memory.
      m_MainPart.uploadsInMemory( true );
   }
   // For prudence,

   bool bDummy = false;
   if ( ! m_MainPart.parseBody( input, bDummy ) )
   {
      m_posts->insert(
               FALCON_GC_HANDLE(&(new String(":error"))->bufferize()),
               FALCON_GC_HANDLE(&(new String( m_MainPart.error() ))->bufferize())
          );
      return false;
   }

   // shouldn't be necessary as we started the upload in memory mode.
   m_MainPart.closeUpload();

   // it's a singlepart or multipart?
   PartHandler* child = m_MainPart.child();
   if( child == 0 )
   {
      // parse the post fields
      String post_data;
      m_MainPart.getMemoryData( post_data );
      Falcon::WOPI::Utils::parseQuery( post_data, *m_posts );
   }
   /*
   else
   {
      while( child != 0 )
      {
         addUploaded( child );
         child = child->next();
      }
   }
   */

   return true;
}


bool Request::getField( const String& fname, String& value ) const
{
   Item temp;
   if( ! getField( fname, temp ) || ! temp.isString() )
      return false;

   value = *temp.asString();
   return true;
}


bool Request::getField( const String& fname, Item& value ) const
{
   Item* res;
   // try in gets.
   if( (res = m_gets->find( fname )) == 0 )
   {
      if( (res = m_posts->find(fname) ) == 0 )
      {
         if( (res = m_cookies->find(fname) ) == 0 )
         {
            return false;
         }
      }
   }

   value = *res;
   return true;
}


void Request::fwdGet( String& fwd, bool all ) const
{
   forward( *m_gets, *m_posts, fwd, all );
}


void Request::fwdPost( String& fwd, bool all ) const
{
   Utils::dictAsInputFields( fwd, *m_posts );
   if( all )
   {
      Utils::dictAsInputFields( fwd, *m_gets );
   }
}

void Request::forward( const ItemDict& main, const ItemDict& aux, String& fwd, bool all ) const
{
   fwd.size(0);
   Utils::fieldsToUriQuery( main, fwd );
   if( all )
   {
      Utils::fieldsToUriQuery( aux, fwd );
   }
}


bool Request::setURI( const String& uri )
{
   if( m_uri.parse( uri ) )
   {
      m_sUri = uri;
      if ( m_uri.query().size() != 0 )
      {
         Utils::parseQuery( m_uri.query(), *m_gets );
      }

      m_location = m_uri.path();

      return true;
   }

   return false;
}


//================================================================
// Falcon interface.
//


void Request::addUploaded( PartHandler* ph, const String& prefix )
{
   String key = prefix.size() == 0 ? ph->name(): prefix + "." + ph->name();

   if( ph->isFile() )
   {
      // an empty file field?
      if( ph->filename().size() == 0 )
      {
         // puts a nil item.
         Falcon::WOPI::Utils::addQueryVariable( key, Item(), *m_base->m_posts );
      }
      else
      {
         // configure the part
         CoreObject* upld = m_upld_c->createInstance();
         upld->setProperty( "mimeType", SafeItem( new CoreString(ph->contentType())) );
         upld->setProperty( "filename", SafeItem( new CoreString(ph->filename())) );
         upld->setProperty( "size", ph->uploadedSize() );

         if( ph->error().size() != 0 )
         {
            // was there an error?
            upld->setProperty( "error", SafeItem( new CoreString(ph->error())) );
         }
         else
         {
            // no? -- store the data or the temporary file name.
            if ( ph->isMemory() )
            {
               MemBuf* mb = new MemBuf_1(0);
               ph->getMemoryData( *mb );
               upld->setProperty( "data", SafeItem(mb) );
            }
            else
            {
               upld->setProperty( "storage", SafeItem( new CoreString(ph->tempFile())) );
            }
         }

         // It may take some time before we can reach the vm,
         // so it's better to be sure we're not marked.
         Falcon::WOPI::Utils::addQueryVariable( key, SafeItem(upld), m_base->m_posts->items() );
      }
   }
   else
   {
      // We have just to add this field.
      if( ph->isMemory() )
      {
         String temp;
         CoreString* value = new CoreString;
         ph->getMemoryData( temp );
         temp.c_ize();
         value->fromUTF8( (char *) temp.getRawStorage() );

         // It may take some time before we can reach the vm,
         // so it's better to be sure we're not marked.
         Falcon::WOPI::Utils::addQueryVariable( key, SafeItem(value), m_base->m_posts->items() );
      }
      // else, don't know what to do
   }

   PartHandler* child = ph->child();
   while( child != 0 )
   {
      addUploaded( child, key );
      child = child->next();
   }
}


bool CoreRequest::processMultiPartBody()
{
   PartHandler* child = m_base->m_MainPart.child();
   if( child == 0 )
   {
      return false;
   }

   while( child != 0 )
   {
      addUploaded( child );
      child = child->next();
   }

   return true;
}


void CoreRequest::configFromModule( const Module* mod )
{
   AttribMap* attribs = mod->attributes();
   if( attribs == 0 )
   {
      return;
   }

   VarDef* value = attribs->findAttrib( FALCON_WOPI_MAXMEMUPLOAD_ATTRIB );
   if( value != 0 && value->isInteger() )
   {
      m_base->m_nMaxMemUpload = value->asInteger();
   }

   value = attribs->findAttrib( FALCON_WOPI_TEMPDIR_ATTRIB );
   if( value != 0 && value->isString() )
   {
      m_base->m_sTempPath.bufferize( *value->asString() );
   }
}


//================================================================
// Override
//

CoreObject *CoreRequest::clone() const
{
   // request object is not clone able.
   return 0;
}



bool CoreRequest::setProperty( const String &prop, const Item &value )
{
   if ( m_bPostInit )
   {
      postInit();
      m_bPostInit = false;
   }

   if( prop == "sidField" )
   {
      if( value.isString() )
         m_base->m_sSessionField.bufferize(*value.asString());
      else
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra("sidField") );
   }
   else if ( prop == "autoSession" )
   {
      m_bAutoSession = value.isTrue();
   }
   else
   {
      readOnlyError( prop );
   }
   return true;
}

bool CoreRequest::getProperty( const String &prop, Item &value ) const
{
   if ( m_bPostInit )
   {
      const_cast<CoreRequest*>(this)->postInit();
      const_cast<CoreRequest*>(this)->m_bPostInit = false;
   }

   if( prop == "gets" )
   {
      value = m_base->m_gets;
   }
   else if( prop == "posts" )
   {
      value = m_base->m_posts;
   }
   else if( prop == "cookies" )
   {
      value = m_base->m_cookies;
   }
   else if( prop == "headers" )
   {
      value = m_base->m_headers;
   }
   else if( prop == "parsed_uri" )
   {
      value = Utils::makeURI( m_base->m_uri );
   }
   else if( prop == "protocol" )
   {
      value = m_base->m_protocol;
   }
   else if( prop == "request_time" )
   {
      value = m_base->m_request_time;
   }
   else if( prop == "bytes_sent" )
   {
      value = m_base->m_bytes_sent;
   }
   else if( prop == "content_length" )
   {
      value = m_base->m_content_length;
   }
   else if( prop == "method" )
   {
      value = m_base->m_method;
   }
   else if( prop == "content_type" )
   {
      value = m_base->m_content_type;
   }
   else if( prop == "content_encoding" )
   {
      value = m_base->m_content_encoding;
   }
   else if( prop == "ap_auth_type" )
   {
      value = m_base->m_ap_auth_type;
   }
   else if( prop == "user" )
   {
      value = m_base->m_user;
   }
   else if( prop == "location" )
   {
      value = m_base->m_location;
   }
   else if( prop == "uri" )
   {
      value = m_base->m_sUri;
   }
   else if( prop == "filename" )
   {
      value = m_base->m_filename;
   }
   else if( prop == "path_info" )
   {
      value = m_base->m_path_info;
   }
   else if( prop == "args" )
   {
      value = m_base->m_args;
   }
   else if( prop == "remote_ip" )
   {
      value = m_base->m_remote_ip;
   }
   else if( prop == "sidField" )
   {
      value = m_base->m_sSessionField;
   }
   else if( prop == "startedAt" )
   {
      value = m_base->startedAt();
   }
   else if( prop == "provider" )
   {
      value = m_provider;
   }
   else if( prop == "autoSession" )
   {
      value.setBoolean( m_bAutoSession );
   }
   else
   {
      return defaultProperty( prop, value );
   }

   return true;
}


void Request::gcMark( uint32 mark )
{
   if( mark != m_mark )
   {
      m_mark = mark;
      m_gets->gcMark( mark );
      m_posts->gcMark( mark );
      m_cookies->gcMark( mark );
      m_headers->gcMark( mark );
   }
}

}
}

/* end of request.cpp */

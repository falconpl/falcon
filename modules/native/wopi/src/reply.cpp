/*
   FALCON - The Falcon Programming Language.
   FILE: reply.cpp

   Web Oriented Programming Interface

   Object encapsulating reply.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 19 Feb 2010 19:30:26 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/wopi/reply.h>
#include <falcon/error.h>
#include <falcon/wopi/modulewopi.h>

namespace Falcon {
namespace WOPI {

CookieParams::CookieParams():
   m_expire(0),
   m_max_age(-1),
   m_version(1),
   m_secure(false),
   m_httpOnly( false ),
   m_bValueGiven( false )
{
}


Reply::Reply( ModuleWopi* wopi ):
   m_nStatus( FALCON_WOPI_DEFAULT_REPLY_STATUS ),
   m_sReason( FALCON_WOPI_DEFAULT_REPLY_REASON ),
   m_bHeadersSent( false )
{
   // prepare default values
   setContentType( "text/html; charset=utf-8" );
   setHeader( "Pragma", "no-cache" );
   setHeader( "Cache-Control", "no-cache" );

   // and THEN tell we're using the defaults.
   m_bDefaultContent = true;

   m_module = wopi;
   m_commitHandler = 0;
}

Reply::~Reply()
{
   delete m_commitHandler;
}


void Reply::clearCookie( const String& cname )
{
   String sCookie;

   Falcon::URI::URLEncode( cname, sCookie );
   sCookie.append( "=;Max-Age=0;expires=0" );
   
   m_mCookies[ cname ] = sCookie;
}



bool Reply::setCookie( const String& cname, const CookieParams &p )
{
   String sCookie;

   Falcon::URI::URLEncode( cname, sCookie );
   sCookie.append( "=" );

   // if value is not a string, stringify it
   if ( p.m_bValueGiven )
   {
     Falcon::String temp;
     //URLEncode will encode also quotes, so that quoted values are safe.
     Falcon::URI::URLEncode( p.m_value, temp );
     sCookie += temp;
   }

   // Expire part
   if ( p.m_expire != 0 )
   {
      String sDummy;
      p.m_expire->toRFC2822( sDummy );
      sCookie += "; expires=" + sDummy;
   }
   else if ( p.m_expire_string.size() != 0 )
   {
      sCookie += "; expires=" + p.m_expire_string;
   }
   else if( p.m_max_age >= 0 )
   {
      sCookie.A("; Max-Age=").N( p.m_max_age );
   }

   // path part
   if ( p.m_path.size() != 0 )
   {
      sCookie += "; Path=" + p.m_path;
   }

   if ( p.m_domain.size() != 0 )
   {
      sCookie += "; Domain=" + p.m_domain;
   }

   if ( p.m_version != 0 )
   {
     sCookie.A( "; Version=" ).N( p.m_version );
   }

   if ( p.m_secure )
   {
      sCookie += "; Secure";
   }

   if ( p.m_httpOnly )
   {
      sCookie += "; httponly";
   }

   m_mCookies[ cname ] = sCookie;
   return true;
}


bool Reply::setHeader( const String& fname, const String& value )
{
   if( m_bHeadersSent )
      return false;

   // Find the header.
   Utils::StringMap::iterator ifield = m_mHeaders.find( fname );
   if( ifield != m_mHeaders.end() )
   {
      ifield->second = value;
   }
   else
   {
      m_mHeaders[ fname ] = value;
   }

   return true;
}

bool Reply::removeHeader( const String& fname )
{
   if( m_bHeadersSent )
      return false;

   // Find the header.
   Utils::StringMap::iterator ifield = m_mHeaders.find( fname );
   if( ifield != m_mHeaders.end() )
   {
      m_mHeaders.erase( ifield );
      return true;
   }

   return false;
}

bool Reply::getHeader( const String& fname, String& value )
{
   // Find the header.
   Utils::StringMap::iterator ifield = m_mHeaders.find( fname );
   if( ifield != m_mHeaders.end() )
   {
      value = ifield->second;
      return true;
   }

   return false;
}

ItemDict* Reply::getHeaders()
{
   ItemDict* ld = new ItemDict;

   Utils::StringMap::iterator ifield = m_mHeaders.begin();
   while( ifield != m_mHeaders.end() )
   {
      const String& key = ifield->first;
      const String& value = ifield->second;

      ld->insert( FALCON_GC_HANDLE( new String( key ) ), FALCON_GC_HANDLE( new String( value ) ) );
      ++ifield;
   }

   return ld;
}


bool Reply::setContentType( const String& type )
{
   if( m_bHeadersSent )
      return false;

   m_bDefaultContent = false;
   setHeader( "Content-Type", type );

   // do the type includes an encoding?
   uint32 pos = type.find( "charset" );
   if( pos != String::npos )
   {
      uint32 p1 = type.find("=", pos);
      if( p1 != String::npos )
      {
         uint32 p2 = type.find(";", p1 );
         // ok also if p2 is npos
         m_sEncoding = type.subString( p1+1, p2 );
         m_sEncoding.trim();
         return true;
      }
   }

   // else default the encoding to C
   m_sEncoding = "C";
   return true;
}

bool Reply::setContentType( const String& type, const String& subtype )
{
   if( m_bHeadersSent )
      return false;

   m_bDefaultContent = false;
   setHeader( "Content-Type", type + "/" + subtype );
   m_sEncoding = "C";
   return true;
}


bool Reply::setContentType( const String& type, const String& subtype, const String& encoding )
{
   if( m_bHeadersSent )
      return false;

   m_bDefaultContent = false;
   setHeader( "Content-Type", type + "/" + subtype + "; charset=" + encoding );
   m_sEncoding = encoding;

   return true;
}


bool Reply::setRedirect( const String& url, uint32 timeout )
{
   if( m_bHeadersSent )
      return false;

   String dest;
   dest.N( (int64)timeout ).A( "; url=" ).A( url );

   setHeader( "Refresh", dest );
   return true;
}



bool Reply::commit()
{
   // already sent -- reuturn false.
   if ( m_bHeadersSent )
      return false;

   // prepare the headers
   if( m_commitHandler != 0 )
   {
      m_commitHandler->startCommit(this);

      Utils::StringMap::const_iterator ic = m_mHeaders.begin();
      while( ic != m_mHeaders.end() )
      {
         //else -- raise error?
         m_commitHandler->commitHeader( this, ic->first, ic->second );
         ++ic;
      }

      ic = m_mCookies.begin();
      while( ic != m_mCookies.end() )
      {
         m_commitHandler->commitHeader( this, "Set-Cookie", ic->second );
         ++ic;
      }

      m_commitHandler->endCommit( this );
   }

   m_bHeadersSent = true;
   return true;
}


void Reply::gcMark( uint32 mark )
{
   m_module->gcMark( mark );
}


void Reply::setCommitHandler( CommitHandler* h )
{
   delete m_commitHandler;
   m_commitHandler = h;
}


}
}

/* end of reply.cpp */

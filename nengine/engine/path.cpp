/*
   FALCON - The Falcon Programming Language.
   FILE: path.cpp

   RFC 3986 compliant path-parth - implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Feb 2008 22:05:33 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/path.cpp"

#include <falcon/uri.h>
#include <falcon/path.h>

namespace Falcon
{

Path::Path():
   m_bValid( true )
{
}

void Path::copy( const Path &other )
{
   m_encoded = other.m_encoded;
   m_filename = other.m_filename;
   
   m_bValid = other.m_bValid;

   m_device = other.m_device;
   m_location = other.m_location;
   m_file = other.m_file;
   m_extension = other.m_extension;
}


void Path::clear()
{
   m_encoded.size(0);
   m_filename.size(0);
   
   m_bValid = true;

   m_device.size(0);
   m_location.size(0);
   m_file.size(0);
   m_extension.size(0);
}


void Path::encode_internal() const
{
   if( m_device.size() != 0 )
      m_encoded = "/" + m_device + ":" + m_location;
   else
      m_encoded = m_location;

   if ( m_extension.size() > 0 )
   {
      m_filename = m_file + "." + m_extension; 
   }
   else
   {
      m_filename = m_file;
   }
   
   if( m_encoded.size() > 0 && m_encoded.getCharAt( m_encoded.length() - 1 ) != ':' )
   {
      // it's right for a location to terminate with /
      m_encoded += "/";
   }
   
   m_encoded += m_filename;
}


bool Path::parse( const String &path )
{
   clear();
   
   uint32 len = path.length();
   if ( len == 0 )
   {
      return true;
   }

   // a single element should be considered as the file.
   bool bColon = false;
   uint32 p = 0;
   while( p < len )
   {
      uint32 chr = path.getCharAt( p );
      if ( chr == '\\' )
      {
         chr = '/';
      }

      switch( chr )
      {
         case ':':
            if ( bColon )
            {
               m_bValid = false;
               return false;
            }

            bColon = true;
            m_location.size(0); // prevent storing the intial "/"
            m_device = m_filename;
            m_filename.size(0);
            break;

         case '/':
            if ( m_filename.size() == 0 )
            {
               if( m_location.size() == 0 )
                  m_location = "/";

               // otherwise ignore; treat // as just /
            }
            else {
               if ( m_location.size() != 0 && m_location.getCharAt( m_location.length() - 1 ) != '/' )
                  m_location.append( '/' );
               m_location += m_filename;
               m_filename.size(0);
            }
            break;

         default:
            m_filename.append( chr );
      }

      p++;
   }

   // detect if we have an extension
   uint32 pos = m_filename.rfind( '.' );
   if ( pos > 0 && pos != String::npos )
   {
      m_extension = m_filename.subString( pos + 1 );
      m_file = m_filename.subString(0, pos);
   }
   else
   {
      m_file = m_filename;
   }

   m_bValid = true;
   return true;
}


bool Path::resource( const String &res )
{
   if ( res != m_device )
   {
      if ( res.findFirstOf( ":;/\\" ) != String::npos )
      {
         return false;
      }
      
      m_encoded.size(0);
      m_device = res;
   }
   
   return true;
}


bool Path::location( const String &loc )
{
   if( m_location != loc )
   {
      uint32 pos = loc.findFirstOf( ":;" );
      if ( pos != String::npos )
      {
         return false;
      }
      
      m_location = loc;
      m_encoded.size(0);

      uint32 pos1 = m_location.find( "\\" );
      while( pos1 != String::npos )
      {
         m_location.setCharAt( pos1, '/' );
         pos1 = m_location.find( "\\", pos1 );
      }

      // remove trailing "/"
      if ( m_location.size() > 1 && m_location.getCharAt( m_location.length() - 1 ) == '/' )
         m_location.remove( m_location.length() - 1, 1 );
   }
   
   return true;
}

bool Path::file( const String &file )
{
   if ( m_file != file )
   {
  
      if ( file.findFirstOf( "/\\.:;" ) != String::npos )
      {
         return false;
      }

      m_encoded.size(0);
      m_file = file;
   }
   
   return true;
}


bool Path::ext( const String &ext )
{
   if ( m_extension != ext )
   {
      
      if ( ext.findFirstOf( "/\\:;." ) != String::npos )
      {
         return false;
      }
      
      m_extension = ext;
      m_encoded.size(0);
   }
   
   return true;
}


bool Path::fileext( const String &fname )
{
   if ( fname.findFirstOf( "/\\:;" ) != String::npos )
   {
      return false;
   }
   
   uint32 posdot = fname.rfind( '.' );
   if ( posdot != String::npos && posdot != 0 && posdot != fname.length() - 1 )
   {
      m_file = fname.subString( 0, posdot );
      m_extension = fname.subString( posdot + 1 );
   }
   else {
      m_file = fname;
      m_extension = "";
   }
   
   m_filename = fname;
   m_encoded.size(0);
   return true;
}


//==================================================
// Utils
// 


void Path::getWinFormat( String &str ) const
{
   encode();
   
   str.reserve( m_encoded.size() );

   // if we have a resource specifier, we know we have a leading /
   uint32 startPos = m_encoded.getCharAt(0) == '/' && m_device.size() != 0 ? 1: 0;
   uint32 endPos = m_encoded.length();
   while( startPos < endPos )
   {
      uint32 chr = m_encoded.getCharAt( startPos );
      if( chr != '/' )
         str.append( chr );
      else
         str.append( '\\' );
      ++startPos;
   }
}


bool Path::getFullLocation( String &str ) const
{
   if( m_device.size() != 0 )
      str = "/" + m_device + ":" + m_location;
   else
      str = m_location;

   return str.size() != 0;
}


bool Path::setFullLocation( const String &str )
{
   length_t pos = str.find( ':' );
   if( pos == str.length() )
   {
      if( location( str ) )
      {
         m_device = "";
         return true;
      }
   }
   else
   {
      if( resource( str.subString(0,pos) ) && location( str.subString( pos + 1 ) ) )
      {
         return true;
      }
   }
   
   return false;
}


bool Path::getWinLocation( String &str ) const
{
   str = location();
   if( str == "" )
   {
      return false;
   }   

   uint32 len = str.length();
   for( uint32 i = 0; i < len; i ++ )
   {
      if ( str.getCharAt( i ) == '/' )
         str.setCharAt( i, '\\' );
   }

   return true;
}


bool Path::getFullWinLocation( String &str ) const
{
   if ( m_device != "" )
   {
      String loc;
      if ( getWinLocation( loc ) )
      {
         str = m_device + ":" + loc;
         return true;
      }
   }
   else if ( getWinLocation( str ) )
   {
      return true;
   }
   
   // str has been cleaned by getWinLocation()
   return false;
}



bool Path::extendLocation( const String &npath )
{
   if ( npath.find(':') != String::npos )
   {
      return false;
   }
   
   if ( m_location.size() == 0 )
      m_location = npath;
   else
   {
      if( npath.size() != 0 )
      {
         if( npath.getCharAt(0) != '/' )
            m_location += "/";
         m_location += npath;
      }
   }

   // remove trailing "/".
   if ( npath.size() > 0 && m_location.getCharAt( m_location.length() - 1 ) == '/' )
      m_location.remove( m_location.length() - 1, 1 );

   m_encoded.size(0);
   return true;
}


bool Path::isAbsolute() const
{
   return m_location.size() > 0 && m_location.getCharAt(0) == '/';
}


bool Path::isLocation() const
{
   return (m_file.size() == m_extension.size()) == 0;
}


void Path::split( String &loc, String &name, String &ext )
{
   if ( m_device.size() > 0 )
   {
      loc = m_device + ":" + m_location;
   }
   else
      loc = m_location;

   name = m_file;
   ext = m_extension;
}


void Path::split( String &res, String &loc, String &name, String &ext )
{
   res = m_device;
   loc = m_location;
   name = m_file;
   ext = m_extension;
}


void Path::splitWinFormat( String &res, String &loc, String &name, String &ext )
{
   split( res, loc, name, ext );
   uint32 len = loc.length();
   for( uint32 i = 0; i < len; i ++ )
   {
      if ( loc.getCharAt( i ) == '/' )
         loc.setCharAt( i, '\\' );
   }
}


void Path::join( const String &loc, const String &name, const String &ext )
{
   m_encoded = loc;
   if( loc.length() !=  0 )
   {
      if( loc.getCharAt( loc.length() - 1 )  != '/' )
         m_encoded += '/';
   }

   if ( name.length() != 0 )
   {
      m_encoded += name;

      if( ext.length() != 0 )
      {
         if( name.getCharAt( name.length() - 1 ) != '.' )
            m_encoded.append( '.' );
         m_encoded += ext;
      }
   }

   parse( m_encoded );
}


void Path::join( const String &res, const String &loc, const String &name, const String &ext )
{
   m_encoded = res;

   if ( res.length() != 0 && res.getCharAt( res.length() - 1 ) != ':' )
      m_encoded.append( ':' );

   m_encoded += loc;

   if( loc.length() !=  0 )
   {
      if( loc.getCharAt( loc.length() - 1 )  != '/' )
         m_encoded += '/';
   }

   if ( name.length() != 0 )
   {
      m_encoded += name;

      if( ext.length() != 0 )
      {
         if( name.getCharAt( name.length() - 1 ) != '.' )
            m_encoded.append( '.' );
         m_encoded += ext;
      }
   }

   parse(m_encoded);
}


void Path::winToUri( String &ret )
{
   int size = ret.length();
   ret.bufferize();
   bool prefix = false;

   for( int i = 0; i < size; i ++ ) {
      int chr = ret.getCharAt( i );
      if( chr == '\\' )
         ret.setCharAt( i, '/' );
      else if ( chr == ':' )
         prefix = true;
   }

   if( prefix && ret.getCharAt(0) != '/' )
      ret.prepend('/');
}


void Path::uriToWin( String &result )
{
   result.bufferize();
   bool bRem = false;

   unsigned int i = 0;
   while( i < result.length() )
   {
      int chr = result.getCharAt( i );
      switch ( chr )
      {
      case '/':
         // "/C:" disk specificator?
         if ( i == 0 )
            bRem = true;
         else
            bRem = false;

         result.setCharAt( i, '\\' );
         break;
      
      case ':':
         // was the first "/" to be removed?
         if ( bRem )
         {
            result.remove(0,1);
            // get I back.
            i--;
         }
         break;
      
      case '+':
         result.setCharAt( i, ' ' );
         break;
      
      // hex escape?
      case '%':
         {
            // have we got enough space for 2 chars?
            if( result.length() < i + 2 )
            {
               return;
            }

            char n1 = result.getCharAt( i+1 );
            char n2 = result.getCharAt( i+2 );
            uint32 r = 0;
            if( n1 >= 'a' && n1 <= 'f' ) r += (n1-'a')*0x10;
            else if( n1 >= 'A' && n1 <= 'F' ) r += (n1-'A')*0x10;
            else if( n1 >= '0' && n1 <= '9' ) r += (n1-'0')*0x10;

            if( n2 >= 'a' && n2 <= 'f' ) r += (n2-'a');
            else if( n1 >= 'A' && n2 <= 'F' ) r += (n2-'A');
            else if( n2 >= '0' && n2 <= '9' ) r += (n2-'0');

            // remove extra chars
            result.remove( i, i+2 );
            // change the transformed one.
            result.setCharAt( i, r );
         }
         break;
      }

      i ++;
   }
}


}

/* end of path.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: uri.cpp

   RFC 3986 - Uniform Resource Identifier - implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2008 12:23:28 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/uri.h>
#include <falcon/memory.h>
#include <falcon/autocstring.h>

namespace Falcon
{

Path::Path():
   m_resEnd(String::npos),
   m_pathStart(String::npos),
   m_pathEnd(String::npos),
   m_fileStart(String::npos),
   m_fileEnd(String::npos),
   m_extStart(String::npos),
   m_bValid( true )
{
}


Path::Path( const Path &other )
{
   m_path = other.m_path;
   m_bValid = other.m_bValid;

   m_resEnd = other.m_resEnd;
   m_pathStart = other.m_pathStart;
   m_pathEnd = other.m_pathEnd;
   m_fileStart = other.m_fileStart;
   m_fileEnd = other.m_fileEnd;
   m_extStart = other.m_extStart;
}

void Path::set( const String &p )
{
   m_path = p;
   analyze( false );
}

bool Path::analyze( bool isWin )
{
   // reset counters
   m_resEnd = String::npos;
   m_pathStart = String::npos;
   m_pathEnd = String::npos;
   m_fileStart = String::npos;
   m_fileEnd = String::npos;
   m_extStart = String::npos;

   uint32 len = m_path.length();

   if ( len == 0 )
   {
      m_bValid = true;
      return true;
   }

   // a single element should be considered as the file.
   m_fileStart = 0;
   uint32 p = 0;
   bool bHadColon = false;

   while( p < len )
   {
      uint32 chr = m_path.getCharAt( p );

      if ( chr == ':' )
      {
         // double colon -- error ?
         // error also if this was the first loop
         // also, if we had a "/" before, this is an error
         if ( bHadColon || p == 0 ||
            ( m_pathStart != String::npos && m_pathStart != 0 ) )
         {
            m_bValid = false;
            return false;
         }

         // shall we add a slash?
         if ( m_path.getCharAt( 0 ) != '/' )
         {
            p++;
            m_path.prepend( '/' );
         }

         m_resEnd = p;

         // reset other starts, just in case.
         m_pathStart = String::npos;
         m_pathEnd = String::npos;
         m_fileStart = String::npos;
         m_fileEnd = String::npos;
         m_extStart = String::npos;
      }
      else if ( (chr == '\\' && isWin) || chr == '/' )
      {
         m_path.setCharAt( p, '/' );
         // first !
         if ( m_pathStart == String::npos )
         {
            // path is from beginning or from after ":"
            // (which may be also from here, but it's ok)
            if ( m_resEnd != String::npos )
               m_pathStart = m_resEnd + 1;
            else
               m_pathStart = 0;
         }
         m_pathEnd = p;

         // make the file to start from here.
         m_fileStart = m_pathEnd + 1;
         m_extStart = String::npos;
      }
      else if ( chr == '.' )
      {
         // skip initial "."s, as they don't make an extension.
         if ( m_fileStart < p && p > 0  && m_path.getCharAt( p - 1 ) != '.' )
         {
            m_fileEnd = p;
            m_extStart = p + 1;
         }
      }

      ++p;
   }

   m_bValid = true;
   return true;
}

void Path::setFromWinFormat( const String &p )
{
   m_path = p;
   analyze( true );
}


void Path::getWinFormat( String &str ) const
{
   str.size(0);
   str.reserve( m_path.size() );

   // if we have a resource specifier, we know we have a leading /
   uint32 startPos = (m_pathStart != String::npos && m_pathStart > 0) ? 1 : 0;
   uint32 endPos = m_path.length();
   while( startPos < endPos )
   {
      uint32 chr = m_path.getCharAt( startPos );
      if( chr != '/' )
         str.append( chr );
      else
         str.append( '\\' );
   }
}

bool Path::getResource( String &str ) const
{
   if ( m_resEnd != String::npos )
   {
      str = m_path.subString( 1, m_resEnd );
      return true;
   }

   return false;
}


bool Path::getLocation( String &str ) const
{
   if ( m_pathStart != String::npos )
   {
      // m_pathStart is "/" and m_fileStart is one after "/"
      str = m_path.subString( m_pathStart, m_pathEnd );
      return true;
   }

   str.size(0);
   return false;
}


bool Path::getWinLocation( String &str ) const
{
   if ( ! getLocation( str ) )
      return false;

   uint32 len = str.length();
   for( uint32 i = 0; i < len; i ++ )
   {
      if ( str.getCharAt( i ) == '/' )
         str.setCharAt( i, '\\' );
   }

   return true;
}


bool Path::getFile( String &str ) const
{
   if ( m_fileStart != String::npos )
   {
      str = m_path.subString( m_fileStart, m_fileEnd );
      return true;
   }

   str.size(0);
   return false;
}

bool Path::getFilename( String &str ) const
{
   if ( m_fileStart != String::npos )
   {
      str = m_path.subString( m_fileStart );
      return true;
   }

   str.size(0);
   return false;
}



bool Path::getExtension( String &str ) const
{
   if ( m_extStart != String::npos )
   {
      str = m_path.subString( m_extStart );
      return true;
   }

   str.size(0);
   return false;
}


void Path::setResource( const String &res )
{
   if ( res.size() )
   {
      if ( m_resEnd != String::npos )
      {
         m_path.change( 1, m_resEnd, res );
      }
      else {
         m_path.prepend( "/" + res );
      }
   }
   else {
      if ( m_resEnd != String::npos )
      {
         m_path.change( 0, m_resEnd+1, "" );
      }
      // else no need to do nothing
      else
         return;
   }

   analyze( false );
}

void Path::setLocation( const String &in_loc )
{


   if ( in_loc.length() >0 )
   {
      if ( m_pathStart != String::npos )
         m_path.change( m_pathStart, m_pathEnd, in_loc );
      else
      {
         if ( m_resEnd != String::npos )
         {
            if( m_fileStart != String::npos && in_loc.getCharAt( in_loc.length() - 1 ) != '/' )
            {
               String loc = in_loc;
               loc.append( '/' );
               m_path.change( m_resEnd+1, m_resEnd+1, loc );
            }
            else
               m_path.change( m_resEnd+1, m_resEnd+1, in_loc );
         }
         else
         {
            if ( m_fileStart != String::npos )
            {
               if ( in_loc.getCharAt( in_loc.length() - 1 ) != '/' )
                  m_path.prepend( "/" );
               m_path.prepend( in_loc );
            }
            else
               m_path = in_loc;
         }
      }
   }
   else {
      if( m_pathStart != String::npos )
         m_path.change( m_pathStart, m_pathEnd+1, "" );
      else
         return;
   }

   analyze( false );
}

void Path::setWinLocation( const String &in_loc )
{
   String loc = in_loc;

   if ( loc.length() >0 )
   {
      if ( loc.getCharAt( loc.length() - 1 ) != '\\' )
      {
         loc.append( '\\' );
      }

      if ( m_pathStart != String::npos )
         m_path.change( m_pathStart, m_pathEnd, loc );
      else {
         if ( m_resEnd != String::npos )
         {
            m_path.change( m_resEnd+1, m_resEnd+1, loc );
         }
      }
   }
   else {
      if( m_pathStart != String::npos )
         m_path.change( m_pathStart, m_pathEnd+1, "" );
      else
         return;
   }

   analyze( true );
}


void Path::setFile( const String &file )
{
   if( m_fileStart != String::npos )
   {
      m_path.change( m_fileStart, m_fileEnd, file );
   }
   else {
      // we may loose the extension, but it's ok
      if ( m_pathStart != String::npos )
      {
         if ( file.size() !=  0 )
         {
            if ( m_path.getCharAt( m_path.length() - 1 ) != '/' )
               m_path.append( '/' );
            m_path.append( file );
         }
         else
            m_path.change( m_pathEnd, String::npos, "" );
      }
      else if ( m_resEnd != String::npos )
      {
         m_path.change( m_resEnd + 1, m_resEnd + 1, file );
      }
      else
      {
         m_path.prepend( file );
      }
   }

   analyze( false );
}


void Path::setExtension( const String &extension )
{
   if ( m_extStart != String::npos )
   {
      if( extension.size() != 0 )
         m_path.change( m_extStart, String::npos, extension );
      else
         m_path.change( m_extStart-1, String::npos, "" );

   }
   else {
      if ( extension.size() != 0 )
      {
         if ( m_path.getCharAt( m_path.length() - 1 ) != '.' )
               m_path.append( '.' );
         m_path.append( extension );
      }
      else
         return;
   }

   analyze( false );
}


void Path::setFilename( const String &fname )
{
   if( m_fileStart != String::npos )
   {
      m_path.change( m_fileStart, String::npos, fname );
   }
   else {
      // we may loose the extension, but it's ok
      if ( m_pathStart != String::npos )
      {
         if ( fname.size() !=  0 )
         {
            if ( m_path.getCharAt( m_path.length() - 1 ) != '/' )
               m_path.append( '/' );
            m_path.append( fname );
         }
         else
            m_path.change( m_pathEnd, String::npos, "" );
      }
      else if ( m_resEnd != String::npos )
      {
         m_path.change( m_resEnd + 1, m_resEnd + 1, fname );
      }
      else
      {
         m_path = fname;
      }
   }

   analyze( false );
}

bool Path::isAbsolute() const
{
   if ( m_pathStart != String::npos )
      return m_path.getCharAt( m_pathStart ) == '/';

   return false;
}


bool Path::isLocation() const
{
   return m_fileStart == String::npos;
}


void Path::split( String &loc, String &name, String &ext )
{
   if ( m_pathStart != String::npos )
      loc = m_path.subString( m_pathStart, m_pathEnd );
   else
      loc.size( 0 );

   if ( m_fileStart != String::npos )
      name = m_path.subString( m_fileStart, m_fileEnd );
   else
      name.size( 0 );

   if ( m_extStart != String::npos )
      ext = m_path.subString( m_extStart );
   else
      ext.size( 0 );

}


void Path::split( String &res, String &loc, String &name, String &ext )
{
   if( m_resEnd != String::npos )
      res = m_path.subString( 1, m_resEnd );
   else
      res.size( 0 );

   split( loc, name, ext );
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
   m_path = loc;
   if( loc.length() !=  0 )
   {
      if( loc.getCharAt( loc.length() - 1 )  != '/' && name.length() != 0 )
         m_path += '/';
   }

   if ( name.length() != 0 )
   {
      m_path += name;

      if( ext.length() != 0 )
      {
         if( name.getCharAt( name.length() - 1 ) != '.' )
            m_path.append( '.' );
         m_path += ext;
      }
   }

   analyze( false );
}


void Path::join( const String &res, const String &loc, const String &name, const String &ext, bool bWin )
{
   m_path = res;

   if ( res.length() != 0 && res.getCharAt( res.length() - 1 ) != ':' )
      m_path.append( ':' );

   m_path += loc;

   if( loc.length() !=  0 )
   {
      if( loc.getCharAt( loc.length() - 1 )  != '/' && name.length() != 0 )
         m_path += '/';
   }

   if ( name.length() != 0 )
   {
      m_path += name;

      if( ext.length() != 0 )
      {
         if( name.getCharAt( name.length() - 1 ) != '.' )
            m_path.append( '.' );
         m_path += ext;
      }
   }

   analyze( bWin );
}


//================================================================
// URI
//

URI::URI()
{
   m_queryMap = new Map( &traits::t_string, &traits::t_string );
}

URI::URI( const String &suri )
{
   m_queryMap = new Map( &traits::t_string, &traits::t_string );
   parse( suri );
}

URI::URI( const URI &other )
{
   m_queryMap = new Map( &traits::t_string, &traits::t_string );
   parse( other.m_original );
}

URI::~URI()
{
   delete m_queryMap;
}

void URI::clear()
{
   m_scheme = "";
   m_userInfo = "";
   m_host = "";
   m_port = "";
   m_path.set( "" );
   if ( m_queryMap != 0 )
   {
      delete m_queryMap;
      m_queryMap = 0;
   }
   m_query = "";
   m_fragment = "";
   m_encoded = "";
   m_bValid = true; // by default.
}

bool URI::parse( const String &newUri, bool decode )
{
   m_bValid = internal_parse( newUri, decode );
   return m_bValid;
}

bool URI::internal_parse( const String &newUri, bool decode )
{
   // had we a previous parsing?
   if ( m_original.size() != 0 )
   {
      clear();
   }
   
   m_original = newUri;

   // We must parse before decoding each element.
   uint32 pStart = 0;
   uint32 pEnd = 0;
   uint32 len = newUri.length();

   typedef enum {
      e_begin,
      e_colon,
      e_postScheme,
      e_host,
      e_port,
      e_path,
      e_done
   } t_status;

   t_status state = e_begin;
   bool bUserGiven = false;
   bool bPortGiven = false;

   String tempPath; // we're setting the path after.

   while( pEnd < len )
   {
      uint32 chr = newUri.getCharAt( pEnd );
      switch ( chr )
      {
         case ':':
            // if we don't have a scheme yet, this is our scheme.
            if( pEnd == 0 )
               return false;
            
            if ( m_scheme.size() == 0 )
               state = e_colon;
            // otherwise, it's just part of what's going on.
         break;

         case '/':
            // Nothing before?
            if ( state == e_begin )
            {
               // we're parsing a (relative or absolute) path and we didn't know!
               state = e_path; 
            }
            // if we had a colon, we have <x>:/
            else if ( state == e_colon )
            {
               if ( pStart == pEnd )  // scheme cannot be empty
                  return false;

               state = e_postScheme; // like begin, we may have a host or a path
               m_scheme = newUri.subString( pStart, pEnd );
               pStart = pEnd+1;
            }
            else if ( state == e_postScheme )
            {
               state = e_host;
               // we have a host starting one next.
               pStart = pEnd + 1;
            }
            else if( state == e_host )
            {
               // we have the full host
               // may be empty (as in file:///path)
               if ( pStart != pEnd )
                  m_host = newUri.subString( pStart, pEnd );
               // anyhow, start the path from here
               pStart = pEnd;
               state = e_path;
            }
            else if ( state == e_port )
            {
               // we have the port.
               if ( pStart == pEnd ) // cannot be empty.
                  return false;

               m_port = newUri.subString( pStart, pEnd );
               // anyhow, start the path from here
               pStart = pEnd;
               state = e_path;
            }
         break;

         case '@':
            // can be found only in host or path state. In path, it is just ignored.
            if ( state == e_host )
            {
               // if we have already user info, we failed.
               if ( bUserGiven )
                  return false;
               
               m_userInfo = newUri.subString( pStart, pEnd );
               pStart = pEnd + 1;
               // state stays host, but signal we have already seen a @ here
               bUserGiven = true;
            }
            else if ( state != e_path )
            {
               return false;
            }
         break;

         case '?':
         case '#':
            // can be found in host, port, path and begin state
            if ( state == e_begin || state == e_path )
            {
               tempPath = newUri.subString( pStart, pEnd );
            }
            else if ( state == e_host )
            {
               m_host = newUri.subString( pStart, pEnd );
            }
            else if ( state == e_port )
            {
               m_port = newUri.subString( pStart, pEnd );
            }
            // cannot be found in e_colon, that would be an error
            else if ( state == e_colon )
               return false;

            // can be found in postScheme state, in which case we have nothing to do
            // in every case, parse the query (+fragment) and exit loop
            if ( chr == '?' )
            {
               if ( ! parseQuery( pEnd + 1, decode ) )
                  return false;
            }
            else {
               if ( ! parseFragment( pEnd + 1 ) )
                  return false;
            }

            // complete loop
            state = e_done;
            pEnd = len;
         break;

         default:
            // if we're in post scheme, a non '/' char starts a path.
            if ( state == e_postScheme )
               state = e_path;
      }
   }

   // what do we have to do now?
   switch ( state )
   {
      case e_begin:
      case e_path:
      case e_colon: // colon too, as it may be i.e. C:
         tempPath = newUri.subString( pStart, pEnd );
      break;

      case e_host:
         m_host = newUri.subString( pStart, pEnd );
         break;

      case e_port:
         m_port = newUri.subString( pStart, pEnd );
         break;
      // in all other cases, just let it through
   }

   if ( decode )
   {
      // decode each element
      if ( m_scheme.size() ) m_scheme = URLDecode( m_scheme );
      if ( m_host.size() ) m_host = URLDecode( m_host );
      if ( m_port.size() ) m_port = URLDecode( m_port );
      if ( tempPath.size() ) tempPath = URLDecode( tempPath );
      if ( m_fragment.size() ) m_fragment = URLDecode( m_fragment );
   }

   // finally, store the path if any
   if ( tempPath.size() )
   {
      m_path.set( tempPath );
      if( m_path.isValid() )
         return false;
   }

   return true;
}


bool URI::parseQuery( uint32 pEnd, bool bDecode )
{
   // break & and = fields.
   uint32 len = m_original.length();
   uint32 pStart = pEnd;

   String tempKey;
   bool bIsValue = false;

   while ( pEnd < len )
   {
      uint32 chr = m_original.getCharAt( pEnd );

      if ( chr == '=' && ! bIsValue )
      {
         // we had the key; we want the value.
         if ( pEnd == pStart )
         {
            // 0 lenght key not allowed
            return false;
         }
         
         if ( bDecode )
            URLDecode( m_original.subString( pStart, pEnd ), tempKey );
         else
            tempKey = m_original.subString( pStart, pEnd );
         bIsValue = true;

         pStart = pEnd + 1;
      }
      else if ( chr  == '&' )
      {
         // have we got a value?
         String val;
         if ( bIsValue && pStart != pEnd )
         {
            
            if ( bDecode )
               URLDecode( m_original.subString( pStart, pEnd ), val );
            else
               val = m_original.subString( pStart, pEnd );
         }

         // save this key
         m_queryMap->insert( &tempKey, &val );
         bIsValue = false;
         pStart = pEnd + 1;
      }
      else if ( chr == '#' )
      {
         return parseFragment( pEnd + 1 );
      }
      
      pEnd ++;
   }

   return true;
}

bool URI::parseFragment( uint32 pos )
{
   // there is actually nothing to do, but getting everything left as substring
   if ( pos < m_original.length() )
      m_fragment = m_original.subString( pos );
   return true;
}


void URI::URLEncode( const String &source, String &target )
{
   target = ""; // resets manipulator
   target.reserve( source.size() );

   // encode as UTF-8
   AutoCString sutf( source );
   const char *cutf = sutf.c_str();
   target.reserve( sutf.length() );

   while ( *cutf != 0 )
   {
      unsigned char chr = (unsigned char) *cutf;
      
      if ( chr == 0x20 )
      {
         target.append( '+' );
      }
      else if ( chr < 0x20 || chr > 0x7F || isResDelim( chr ) )
      {
         target.append( '%' );
         target.append( URI::CharToHex( chr >> 4 ) );
         target.append( URI::CharToHex( chr & 0xF ) );
      }
      else {
         target.append( chr );
      }

      ++cutf;   
   }
}


bool URI::URLDecode( const String &source, String &target )
{
   // the target buffer can be - at worst - long as the source.
   char *tgbuf = (char *) memAlloc( source.length() + 1 );
   char *pos = tgbuf;
   bool bOk = true;
   
   uint32 len = source.length();
   for( uint32 i = 0; i < len; i ++ )
   {
      uint32 chr = source.getCharAt( i );
      // an URL encoded string cannot have raw characters outside defined ranges.
      if ( chr < 0x20 || chr > 0x7F )
      {
         bOk = false;
         break;
      }

      if ( chr == '+' )
         *pos = ' ';
      else if ( chr == '%' )
      {
         // not enough space?
         if ( i+3 >= len )
         {
            bOk = false;
            break;
         }

         // get the characters -- check also for non-hex digits.
         unsigned char c1, c2;
         if (  ( c1 = HexToChar( source.getCharAt( ++i ) ) ) == 0xFF ||
               ( c2 = HexToChar( source.getCharAt( ++i ) ) ) == 0xFF )
         {
            bOk = false;
            break;
         }

         *pos = c1 << 4 | c2;
      }

      ++pos;
   }

   if ( bOk )
   {
      // reconvert from UTF8 to Falcon
      target.fromUTF8( tgbuf );
   }

   memFree( tgbuf );
   return bOk;
}
  
}
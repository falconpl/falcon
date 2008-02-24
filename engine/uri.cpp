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

};

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
   m_pathStart( String::npos ),
   m_fileStart( String::npos ),
   m_extStart( String::npos )
{
}


Path::Path( const Path &other )
{
   m_path = other.m_path;
   m_pathStart = other.m_pathStart;
   m_fileStart = other.m_fileStart;
   m_extStart = other.m_extStart;
}


void Path::set( const String &p )
{
   m_path = p;
   analyze( false );
}

void Path::analyze( bool isWin )
{
   // reset counters
   m_pathStart = String::npos;
   m_fileStart = String::npos;
   m_extStart = String::npos;

   // start cycle by considering initial slash
   bool bHadSlash = false;
   uint32 p = 0;
   uint32 len = m_path.length();

   if ( len == 0 )
      return;

   uint32 chr = m_path.getCharAt( 0 );

   uint32 firstSlash = String::npos;
   uint32 lastSlash = String::npos;
   uint32 colonPos = String::npos;
   uint32 lastDot = String::npos;

   if ( chr == '/' ) {
      bHadSlash = true;
      firstSlash = 0;
      lastSlash = 0;
      p = 1;
   }
   else {
      bHadSlash = false;
   }

   bool validChar = false;

   while( p < len )
   {
      chr = m_path.getCharAt( p );

      if ( chr == ':' && colonPos == String::npos )
      {
         if ( ! bHadSlash ) {
            p++;
            m_path.prepend( '/' );
            firstSlash = String::npos;
            lastSlash = String::npos;
            // reset slashes.
         }

         colonPos = p;
         validChar = false;
      }
      else if ( (chr == '\\' && isWin) || chr == '/' )
      {
         m_path.setCharAt( p, '/' );
         if ( firstSlash == String::npos )
            firstSlash = p;

         lastSlash = p;
         lastDot = String::npos;
         validChar = false;
      }
      else if ( chr == '.' )
      {
         if ( validChar )
            lastDot = p;
      }
      else {
         validChar = true;
      }

      ++p;
   }

   // analyze our foundings.
   if ( ! colonPos )
   {
      // without a leading colon, location starts from zero, provided we have at least a slash.
      if ( firstSlash != String::npos )
      {
         m_pathStart = 0;
         // file starts right after last slash, which may be the same as first.
         if ( lastSlash < len - 1 )
            m_fileStart = lastSlash + 1;
      }
      // otherwise, the file starts from beginning.
      else
         m_fileStart = 0;
   }
   else {
      m_resEnd = colonPos - 1;
      if ( firstSlash != String::npos )
      {
         m_pathStart = colonPos + 1;
         // file starts right after last slash, which may be the same as first.
         if ( lastSlash < len - 1 )
            m_fileStart = lastSlash + 1;
      }
      else
         m_fileStart = colonPos + 1;
   }

   // extension is after last dot, if present.
   if( lastDot != String::npos )
   {
      if( lastDot < len - 1 )
         m_extStart = lastDot + 1;
   }
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

void Path::getResource( String &str ) const
{
   if ( m_pathStart != String::npos && m_pathStart > 0 )
   {
      // 0 is "/" and m_pathStart-1 is ":"
      str = m_path.subString( 1, m_pathStart - 1 );
   }
   else if ( m_fileStart != String::npos && m_fileStart > 0 )
   {
      str = m_path.subString( 1, m_fileStart - 1 );
   }
   else
      str.size(0);
}


void Path::getLocation( String &str ) const
{
   if ( m_pathStart != String::npos )
   {
      // m_pathStart is "/" and m_fileStart is one after "/" 
      // this is ok also if m_fileStart == npos
      str = m_path.subString( m_pathStart, m_fileStart - 1 );
   }
   else
      str.size(0);

}


void Path::getWinLocation( String &str ) const
{
   getLocation( str );
   uint32 len = str.len();
   for( i = 0; i < len; i ++ )
   {
      if ( str.getCharAt( i ) == '/' )
         str.setCharAt( '\\' );
   }
}


void Path::getFilename( String &str ) const
{
   if ( m_filePos != String::npos )
      str = m_path.subString( m_filePos, m_extPos - 1);
   else
      str.size(0);
}


void Path::getExtension( String &str ) const
{
   if ( m_extPos != String::npos )
      str = m_path.subString( m_extPos );
   else
      str.size(0);
}


void setResource( const String &res )
{
   uint32 start = m_pathStart < m_fileStart ? m_pathStart : m_fileStart;
   if ( start != String::npos && start > 0 )
   {
      m_path.change( 1, start - 1, res );
   }
   else {
      m_path.prepend( "/" + res );
   }
   analyze( false );
}

void setLocation( const String &loc )
{
   if ( m_pathStart != String::npos )
      m_path.change( m_pathStart, m_fileStart-1, loc );
   else 
      m_path.prepend( loc );
}

   /** Sets the location part in MS-Windows format. */
   void setWinLocation( const String &loc );

   /** Sets the file part. */
   void setFile( const String &file );

   /** Sets the extension part. */
   void setExtension( const String &extension );

   /** Returns true if this path is an absolute path. */
   bool isAbsolute() const;
};


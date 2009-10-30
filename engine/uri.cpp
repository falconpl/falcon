/*
   FALCON - The Falcon Programming Language.
   FILE: uri.cpp

   RFC 3986 - Uniform Resource Identifier - implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Feb 2008 12:23:28 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/uri.h>
#include <falcon/memory.h>
#include <falcon/autocstring.h>

namespace Falcon
{

URI::URI():
   m_bValid( true ),
   m_path( this ),
   m_queryMap( 0 )
{
}

URI::URI( const String &suri ):
   m_bValid( true ),
   m_path( this ),
   m_queryMap(0)
{
   parse( suri );
}

URI::URI( const URI &other ):
   m_bValid( true ),
   m_path( this ),
   m_queryMap(0)
{
   parse( other.get( true ) );
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

bool URI::parse( const String &newUri, bool parseQuery, bool decode )
{
   m_bValid = internal_parse( newUri, parseQuery, decode );
   return m_bValid;
}

bool URI::internal_parse( const String &newUri, bool parseQuery, bool decode )
{
   // had we a previous parsing?
   if ( m_original.size() != 0 )
   {
      clear();
   }

   m_original = newUri;
   
   if ( Engine::getWindowsNamesConversion() )
   {
      if ( newUri.find( "\\" ) != String::npos || 
         (newUri.length() > 2 && newUri.getCharAt(1) == ':' ) 
         )
      {
         Path::winToUri( m_original );
      }
   }


   // We must parse before decoding each element.
   uint32 pStart = 0;
   uint32 pEnd = 0;
   uint32 len = m_original.length();

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

   String tempPath; // we're setting the path after.

   while( pEnd < len )
   {
      uint32 chr = m_original.getCharAt( pEnd );
      switch ( chr )
      {
         case ':':
            // if we don't have a scheme yet, this is our scheme.
            if( pEnd == 0 )
               return false;

            if ( state == e_begin )
               state = e_colon;
            else if ( state == e_host )
            {
               m_host = m_original.subString( pStart, pEnd );
               state = e_port;
               pStart = pEnd + 1;
            }
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
               m_scheme = m_original.subString( pStart, pEnd-1 ); // removing extra ':'
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
                  m_host = m_original.subString( pStart, pEnd );
               // anyhow, start the path from here
               pStart = pEnd;
               state = e_path;
            }
            else if ( state == e_port )
            {
               // we have the port.
               if ( pStart == pEnd ) // cannot be empty.
                  return false;

               m_port = m_original.subString( pStart, pEnd );
               // anyhow, start the path from here
               pStart = pEnd;
               state = e_path;
            }
         break;

         case '@':
            // can be found only in host or path state. In path, it is just ignored.
            if ( state == e_port )
            {
               // ops, the host wasn't the host, and the port wasn't the port.
               state = e_host;
               m_userInfo = m_host + ":" + m_original.subString( pStart, pEnd );
               m_host = "";
               pStart = pEnd + 1;
            }
            else if ( state == e_host )
            {
               // if we have already user info, we failed.
               if ( bUserGiven )
                  return false;

               m_userInfo = m_original.subString( pStart, pEnd );
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
               tempPath = m_original.subString( pStart, pEnd );
            }
            else if ( state == e_host )
            {
               m_host = m_original.subString( pStart, pEnd );
            }
            else if ( state == e_port )
            {
               m_port = m_original.subString( pStart, pEnd );
            }
            // cannot be found in e_colon, that would be an error
            else if ( state == e_colon )
               return false;

            // can be found in postScheme state, in which case we have nothing to do
            // in every case, parse the query (+fragment) and exit loop
            if ( chr == '?' )
            {
               if ( ! internal_parseQuery( m_original, pEnd + 1, parseQuery, decode ) )
                  return false;
            }
            else {
               if ( ! internal_parseFragment( pEnd + 1 ) )
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
            else if ( state == e_colon )
            {
               // if we are in colon state, then previous thing (begin) is to be considered host
               m_host = m_original.subString( pStart, pEnd - 1 );
               pStart = pEnd;
               state = e_port;
            }
      }

      pEnd++;
   }

   // what do we have to do now?
   switch ( state )
   {
      case e_begin:
      case e_path:
      case e_colon: // colon too, as it may be i.e. C:
         tempPath = m_original.subString( pStart, pEnd );
      break;

      case e_host:
         m_host = m_original.subString( pStart, pEnd );
         break;

      case e_port:
         m_port = m_original.subString( pStart, pEnd );
         break;
      // in all other cases, just let it through
      default:
         break;
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
      if( ! m_path.isValid() )
         return false;
   }

   return true;
}


bool URI::internal_parseQuery( const String &src, uint32 pEnd, bool parseQuery, bool bDecode )
{
   if ( ! parseQuery )
   {
      uint32 pSharp = src.find( "#", pEnd );
      if( pSharp != String::npos )
      {
         query( src.subString( pEnd, pSharp ) );
         return internal_parseFragment( pSharp+1 );
      }
      else {
         if ( pEnd == 0 )
            query( src );
         else
            query( src.subString( pEnd ) );
      }
      return true;
   }

   delete m_queryMap;
   m_queryMap = new Map( &traits::t_string(), &traits::t_string() );

   // break & and = fields.
   uint32 len = src.length();
   uint32 pStart = pEnd;

   String tempKey;
   bool bIsValue = false;

   while ( pEnd < len )
   {
      uint32 chr = src.getCharAt( pEnd );

      if ( chr == '=' && ! bIsValue )
      {
         // we had the key; we want the value.
         if ( pEnd == pStart )
         {
            // 0 lenght key not allowed
            return false;
         }

         if ( bDecode )
            URLDecode( src.subString( pStart, pEnd ), tempKey );
         else
            tempKey = src.subString( pStart, pEnd );
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
               URLDecode( src.subString( pStart, pEnd ), val );
            else
               val = src.subString( pStart, pEnd );
         }

         // save this key
         m_queryMap->insert( &tempKey, &val );
         bIsValue = false;
         pStart = pEnd + 1;
      }
      else if ( chr == '#' )
      {
         return internal_parseFragment( pEnd + 1 );
      }

      pEnd ++;
   }

   return true;
}

bool URI::parseQuery( bool mode )
{
   m_bValid = internal_parseQuery( m_query, 0, mode, true );
   return m_bValid;
}

bool URI::internal_parseFragment( uint32 pos )
{
   // there is actually nothing to do, but getting everything left as substring
   if ( pos < m_original.length() )
      m_fragment = m_original.subString( pos );
   return true;
}


void URI::query( const String &q, bool encode )
{
   m_encoded = "";

   // ok also if m_queryMap is 0.
   delete m_queryMap;
   m_queryMap = 0;

   if ( encode )
      URLEncode( q, m_query );
   else
      m_query = q;
}



void URI::scheme( const String &s )
{
   m_encoded = "";
   m_scheme = s;
}


void URI::userInfo( const String &s )
{
   m_encoded = "";
   m_userInfo = s;
}


void URI::host( const String &h )
{
   m_encoded = "";
   m_host = h;
}


void URI::port( const String &h )
{
   m_encoded = "";
   m_port = h;
}


void URI::path( const String &p )
{
   m_encoded = "";
   m_path.set( p );
}


void URI::path( const Path &p )
{
   m_encoded = "";
   m_path = p;
}


void URI::fragment( const String &s )
{
   m_encoded = "";
   m_fragment = s;
}


bool URI::hasField( const String &f ) const
{
   if( m_queryMap == 0 )
      return false;

   String *res = (String *) m_queryMap->find( &f );
   return res != 0;
}


bool URI::getField( const String &key, String &value ) const
{
   if( m_queryMap == 0 )
      return false;

   String *res = (String *) m_queryMap->find( &key );
   if ( res != 0 )
   {
      value = *res;
      return true;
   }

   return false;
}


void URI::setField( const String &key, const String &value )
{
   if( m_queryMap == 0 )
   {
      m_queryMap = new Map( &traits::t_string(), &traits::t_string() );
   }

   m_queryMap->insert( &key, &value );
}


bool URI::removeField( const String &key )
{
   if( m_queryMap == 0 )
      return false;

   m_queryMap->erase( &key );
   return true;
}


bool URI::firstField( String &key, String &value )
{
   if ( m_queryMap != 0 && m_queryMap->size() > 0 )
   {
      m_queryIter = m_queryMap->begin();
      key = *(String *) m_queryIter.currentKey();
      value = *(String *) m_queryIter.currentValue();
      return true;
   }

   return false;
}


bool URI::nextField( String &key, String &value )
{
   if ( m_queryMap != 0 && m_queryMap->size() > 0 && m_queryIter.hasNext() )
   {
      m_queryIter.next();
      key = *(String *) m_queryIter.currentKey();
      value = *(String *) m_queryIter.currentValue();
      return true;
   }

   return false;
}


uint32 URI::fieldCount()
{
   if ( m_queryMap != 0 )
      return m_queryMap->size();
   return 0;
}


const String &URI::get( bool synthQuery ) const
{
   if ( m_encoded.size() != 0 )
      return m_encoded;

   if ( synthQuery && m_queryMap != 0 )
      makeQuery();

   if ( m_scheme.size() != 0 )
   {
      m_encoded = m_scheme + ":/";
   }

   if( (m_userInfo.size() != 0) || (m_host.size() != 0) || (m_port.size() != 0) )
   {
      if ( m_encoded.size() != 0 )
         m_encoded += "/";

      if (m_userInfo.size() != 0)
         m_encoded += URLEncode( m_userInfo ) + "@";

      if (m_host.size() != 0)
         m_encoded += URLEncode( m_host );

      if (m_port.size() != 0)
         m_encoded += ":" + URLEncode( m_port );

      if ( m_path.get().size() != 0 )
      {
         if ( ! m_path.isAbsolute() )
            m_encoded += "/";
      }
   }
   else if ( m_scheme.size() != 0 && m_path.isAbsolute() )
      m_encoded += "/";

   if ( m_path.get().size() != 0 )
      m_encoded += URLEncode( m_path.get() );

   if ( m_query.size() != 0 )
      m_encoded += "?" + m_query;

   if ( m_fragment.size() != 0 )
      m_encoded += "#" + URLEncode( m_fragment );

   return m_encoded;
}

const String &URI::makeQuery() const
{
   m_query = "";

   if ( m_queryMap != 0 && m_queryMap->size() > 0 )
   {
      MapIterator iter = m_queryMap->begin();
      m_query += URLEncode( *(String *) iter.currentKey() ) + "=" +
                  URLEncode( *(String *) iter.currentValue() );
      while( iter.hasNext() )
      {
         iter.next();
         m_query += "&";
         m_query += URLEncode( *(String *) iter.currentKey() ) + "=" +
                  URLEncode( *(String *) iter.currentValue() );
      }
   }

   return m_query;
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
      else if ( chr < 0x20 || chr > 0x7F || isSubDelim( chr ) || chr == '%' )
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
         if ( i+3 > len )
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
      else
         *pos = chr;

      ++pos;
   }
   *pos = 0;

   if ( bOk )
   {
      // reconvert from UTF8 to Falcon
      target.fromUTF8( tgbuf );
   }

   memFree( tgbuf );
   return bOk;
}

}

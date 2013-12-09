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
#include <falcon/engine.h>
#include <falcon/autocstring.h>
#include <falcon/path.h>

#include <map>

namespace Falcon
{    

//=========================================
// Authority
//

bool URI::Authority::parse( const String& uriAuth )
{
   m_encoded = "@";
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
   
   String hostPart;
   length_t posat = uriAuth.find( '@' );
   if( posat == String::npos )
   {
      m_user = "";
      m_password = "";
      hostPart = uriAuth;
   }
   else
   {
      String userAuth = uriAuth.subString( 0, posat );
      hostPart = uriAuth.subString( posat+1 );
      length_t poscolon = userAuth.find( ':' );
      if ( poscolon == String::npos )
      {
         if ( ! URLDecode( userAuth, m_user ) )
         {
            return false;
         }
      }
      else
      {
         if( ! URLDecode( userAuth.subString(0,poscolon), m_user )
            || ! URLDecode( userAuth.subString(poscolon+1), m_password ) )
         {
            return false;
         }
      }
   }

   length_t poscolon = hostPart.find( ':' );
   if ( poscolon == String::npos )
   {
      if ( ! URLDecode( hostPart, m_host ) )
      {
         return false;
      }
   }
   else
   {
      if( ! URLDecode( hostPart.subString(0,poscolon), m_host ) 
         || ! URLDecode( hostPart.subString(poscolon+1), m_port ) )
      {
         return false;
      }
   }
   
   return true;
}

void URI::Authority::user( const String& value )
{
   m_encoded = "@";
   m_user = value;
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}


void URI::Authority::clear()
{
   m_encoded = "@";
   m_user.size(0);
   m_password.size(0);
   m_host.size(0);
   m_port.size(0);

   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}

void URI::Authority::password( const String& value  )
{
   m_encoded = "@";
   m_password = value;
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}


void URI::Authority::host( const String& value  )
{
   m_encoded = "@";
   m_host = value;
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}


void URI::Authority::port( const String& value  )
{
   m_encoded = "@";
   m_port = value;
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}


const String& URI::Authority::encode() const
{
   if( m_encoded != "@" )
   {
      return m_encoded;
   }
   
   m_encoded.size(0);
   
   if ( m_user.size() > 0 )
   {
      m_encoded += URLEncode( m_user );
      if( m_password.size() > 0 )
      {
         m_encoded += ":";
         m_encoded += URLEncode( m_password );
      }
      m_encoded += "@";
   }
   
   if ( m_host.size() > 0 )
   {
      m_encoded += URLEncode( m_host );      
   }
   
   if( m_port.size() > 0 )
   {
      m_encoded += ":" + URLEncode( m_port );
   }
   
   return m_encoded;
}


void URI::Authority::ownerURI( URI* owner )
{
   m_owner = owner;
}


void URI::Authority::copy( const Authority& other )
{
   m_user = other.m_user;
   m_password = other.m_password;
   m_host = other.m_host;
   m_port = other.m_port;
   m_encoded = other.m_encoded;

   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}

//=========================================
// Query
//

class URI::Query::Private {
public:
   typedef std::map<String,String> QueryMap;
   QueryMap m_fields;
};


URI::Query::Query( URI* owner ):
   _p( new Private)
{
   m_owner = owner;
}


URI::Query::Query( const Query& other ):
   _p( new Private )
{
   m_owner = 0;
   copy(other);
}


URI::Query::~Query()
{
   delete _p;
}

void URI::Query::copy( const URI::Query& other )
{
   _p->m_fields = other._p->m_fields;
}

bool URI::Query::parse( const String& query )
{
   if( query.size() == 0 )
   {
      m_encoded.size(0);
      return true;
   }
   
   m_encoded = "@";

   // find the next &
   length_t pos = 0;
   length_t posNext;
   
   Private::QueryMap& fields = _p->m_fields;
   
   do
   {
      posNext = query.find( '&', pos );
      
      String part = query.subString( pos, posNext );
      if( part.size() == 0 )
      {
         clear();
         return false;
      }
      
      String key;
      String value;
      length_t eqpos = part.find( '=' );
      if( eqpos == String::npos || eqpos == part.length()-1 )
      {
         value = "";
         if( eqpos != String::npos )
         {
            part = part.subString(0, part.length()-1);
         }

         if( ! URLDecode( part, key ) )
         {
            clear();
            return false;
         }
      }
      else
      {
         if( ! URLDecode( part.subString( 0, eqpos ), key ) 
             || ! URLDecode( part.subString( eqpos+1 ), value ) )
         {
            clear();
            return false;
         }
      }
      
      fields.insert( std::make_pair(key,value) );
      pos = posNext+1;
   }
   while ( posNext != String::npos );
   
   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }

   return true;
}

void URI::Query::ownerURI( URI* owner )
{
   m_owner = owner;
}

const String& URI::Query::encode() const
{
   if( m_encoded != "@" )
   {
      return m_encoded;
   }
   
   m_encoded = "";

   Private::QueryMap::const_iterator iter = _p->m_fields.begin();
   while( iter != _p->m_fields.end() )
   {
      if( m_encoded.size() > 0 )
      {
         m_encoded += "&";
      }
      
      m_encoded += URLEncode( iter->first ) + "=" + URLEncode( iter->second );
      iter++;
   }

   return m_encoded;
}

size_t URI::Query::size() const
{
   return _p->m_fields.size();
}

void URI::Query::put( const String& field, const String& value )
{
   _p->m_fields.insert(std::make_pair(field,value));
   m_encoded = "@";

   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }

}

bool URI::Query::get( const String& field, String& value ) const
{
   Private::QueryMap::const_iterator iter = _p->m_fields.find( field );
   if( iter == _p->m_fields.end() )
   {
      return false;
   }
   
   value = iter->second;
   return true;
}

bool URI::Query::remove( const String &key )
{   
   Private::QueryMap::iterator iter = _p->m_fields.find(key);
   if( iter == _p->m_fields.end() )
   {
      return false;
   }
   
   m_encoded = "@";
   _p->m_fields.erase(iter->second);

   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }

   return true;
}
      

void URI::Query::clear()
{
   m_encoded = "@";
   _p->m_fields.clear();

   if( m_owner != 0 )
   {
      m_owner->invalidate();
   }
}


void URI::Query::enumerateFields( URI::Query::FieldEnumerator& etor ) const
{
   Private::QueryMap::const_iterator iter = _p->m_fields.begin();
   Private::QueryMap::const_iterator end = _p->m_fields.end();
   while( iter != end )
   {
      KeyValue kv( iter->first, iter->second );
      if( ! etor( kv ) )
      {
         break;
      }
      ++iter;
   }
}
     
//=========================================
// URI
//

URI::URI():
   m_bValid(true),
   m_gcMark(0)
{
   m_authority.ownerURI( this );
   m_path.ownerURI( this );
   m_query.ownerURI( this );
}


URI::URI( const URI &other ):
   m_bValid( other.m_bValid ),
   m_scheme( other.m_scheme ), 
   m_authority( other.m_authority ),
   m_path( other.m_path ),
   m_query( other.m_query ),
   m_fragment( other.m_fragment ),
   m_encoded( other.m_encoded ),
   m_gcMark(0)
{
}


URI::~URI()
{
}


void URI::copy( const URI& other )
{
   m_bValid = other.m_bValid;
   m_scheme = other.m_scheme;
   m_authority = other.m_authority;
   m_path = other.m_path;
   m_query = other.m_query;
   m_fragment = other.m_fragment;
   m_encoded = other.m_encoded;
}


const String& URI::encode() const
{
   if( m_encoded != "@" )
   {
      return m_encoded;
   }

   m_encoded = m_scheme;
   if( m_encoded.size() > 0 )
   {
      m_encoded += ":";
   }
   
   if( ! m_authority.empty() )
   {
      m_encoded += "//" + m_authority.encode();
   }
   
   if( ! m_path.empty() )
   {
      if( ! m_authority.empty() && ! m_path.isAbsolute() )
      {
         m_encoded += "/./";  // use ./ to declare the relative path marker
      }
      m_encoded += URLEncodePath(m_path.encode());
   }
   else
   {
      if ( ! m_query.empty() || ! m_fragment.empty() )
      {
         m_encoded += "/";
      }
   }
   
   if( ! m_query.empty() )
   {
      m_encoded += "?" + m_query.encode();
   }
   
   if( m_fragment.size() != 0 )
   {
      m_encoded += "#" + m_fragment;
   }
   
   return m_encoded;
}


bool URI::parse( const String &newUri )
{
   String sAuth, sPath, sQuery;

   // disable change notification
   m_authority.ownerURI(0);
   m_path.ownerURI(0);
   m_query.ownerURI(0);

   if( ! internal_parse( newUri, sAuth, sPath, sQuery ) )
   {
      m_bValid = false;
   }   
   else if ( ! m_authority.parse(sAuth) )
   {
      m_bValid = false;
   }   
   else if ( ! m_path.parse(sPath) )
   {
      m_bValid = false;
   }
   else if(! m_query.parse(sQuery) )
   {
      m_bValid = false;
   }
   else
   {
      m_bValid = true;
   }
   
   // enable change notification
   m_authority.ownerURI(this);
   m_path.ownerURI(this);
   m_query.ownerURI(this);

   return m_bValid;
}
   
void URI::clear()
{
   m_encoded.size(0);

   m_scheme.size(0);
   m_fragment.size(0);
   m_query.clear();
   m_authority.clear();
   m_path.clear();

   m_bValid = true; // by default.
}


bool URI::internal_parse( const String &newUri, String& sAuth, String& sPath, String& sQuery )
{  
   // We must parse before decoding each element.
   uint32 pStart = 0;
   uint32 pEnd = 0;

   typedef enum {
      e_scheme,
      e_auth_slash,
      e_auth_slash2,
      e_auth,
      e_path,
      e_query,
      e_fragment
   } t_status;

   clear();
   
   sAuth.clear();
   sPath.clear();
   sQuery.clear();

   t_status state = e_scheme;
   length_t len = newUri.length();
   while( pEnd < len )
   {
      uint32 chr = newUri.getCharAt( pEnd );
      switch( chr )
      {
         case ':':
            if( state == e_scheme )
            {
               m_scheme = newUri.subString(pStart, pEnd);
               pStart = pEnd+1;
               state = e_auth_slash;
            }
            break;
            
         case '.':
            if( state == e_scheme )
            {
               state = e_path;
            }
            break;
            
         case '/':
            if( state == e_scheme )
            {
               // it's a relative path
               state = e_path;
            }
            else if ( state == e_auth_slash )
            {
               // might be a path or an authority if I get /
               state = e_auth_slash2;
            }
            else if( state == e_auth_slash2 )
            {
               pStart = pEnd+1;
               state = e_auth;
            }
            else if( state == e_auth )
            {
               if( ! URLDecode( newUri.subString(pStart, pEnd), sAuth ) )
               {
                  return false;
               }

               pStart = pEnd;
               state = e_path;
            }                        
            break;

         case '?':
            if( state == e_scheme || state == e_path )
            {
               // preceeding things are a path.
               if( ! URLDecode( newUri.subString(pStart, pEnd ), sPath ) )
               {
                  return false;
               }
               
               pStart = pEnd + 1;
               state = e_query;
            }
            else if( state == e_auth || state == e_auth_slash || state == e_auth_slash2 )
            {
               // no ? accepted in the auth part.
               return false;
            }
            break;
            
            
         case '#':
            if( state == e_scheme || state == e_path )
            {
               // preceeding things are a path.
               if (! URLDecode( newUri.subString( pStart, pEnd ), sPath ) )
               {
                  return false;
               }

               pStart = pEnd + 1;
               state = e_fragment;
            }
            if( state == e_query )
            {
               // preceeding things are a query -- do not decode
               sQuery = newUri.subString( pStart, pEnd );
               pStart = pEnd + 1;
               state = e_fragment;
            }
            else if( state == e_auth || state == e_auth_slash || state == e_auth_slash2 )
            {
               // no # accepted in the auth part.
               return false;
            }
            // there's nothing past fragment.
            pEnd = len;
            break;
      }
      
      ++pEnd;
   }
   
   // what's left open?
   if( pStart < pEnd )
   {
      switch( state )
      {
         // the scheme alone means just path
         case e_scheme: case e_path: case e_auth_slash: case e_auth_slash2:
            URLDecode( newUri.subString( pStart ), sPath );
            break;

         case e_auth:
            URLDecode( newUri.subString( pStart ), sAuth );
            break;

         case e_query:
            sQuery = newUri.subString( pStart );
            break;

         case e_fragment:
            URLDecode( newUri.subString( pStart ), m_fragment );
            break;
      }
   }
   
   m_encoded="@";
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

      if ( ! isUnreserved( chr ) )
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

//TODO: Make one with above
void URI::URLEncodePath( const String &source, String &target )
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
      // in the paths, we can't encode path chars as '/' and '\\'
      else if ( chr < 0x20 || chr > 0x7F || isSubDelim( chr ) ||
            chr == '%'
            || chr == '"' || chr == '\'' || chr == '`'
            || chr == '{' || chr == '}'
            || chr == '<' || chr == '>' )
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
   char *tgbuf = (char *) malloc( source.length() + 1 );
   char *pos = tgbuf;
   bool bOk = true;

   uint32 len = source.length();
   for( uint32 i = 0; i < len; i ++ )
   {
      uint32 chr = source.getCharAt( i );
      // an URL encoded string cannot have raw characters outside defined ranges.
      if ( chr > 0x7F )
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

   free( tgbuf );
   return bOk;
}

}

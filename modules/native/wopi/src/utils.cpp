/*
   FALCON - The Falcon Programming Language.
   FILE: utils.cpp

   Web Oriented Programming Interface (WOPI)

   Utilities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 13 Feb 2010 14:10:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>

#include <falcon/string.h>
#include <falcon/itemdict.h>
#include <falcon/itemarray.h>
#include <falcon/uri.h>
#include <falcon/vm.h>

#include <falcon/wopi/utils.h>

#include <stdlib.h>

#ifdef FALCON_SYSTEM_WIN
#include <windows.h>
#include <time.h>
#else
#include <unistd.h>
#endif


namespace Falcon {
namespace WOPI {
namespace Utils {

void fieldsToUriQuery( const ItemDict& fields, String& target )
{
   class Rator: public ::Falcon::ItemDict::Enumerator
   {

   public:
      Rator( String& tgt ):
         m_target(tgt)
      {}

      virtual ~Rator() {}

      virtual void operator()( const Item& key, Item& value )
      {
         if( m_target.size() != 0 )
         {
            m_target += "&";
         }

         String sName, sValue;
         if( key.isString() )
         {
            URI::URLEncode( *key.asString(), sName );

            if( value.isString() )
            {
               URI::URLEncode( *value.asString(), sValue );
               m_target += sName + "=" + sValue;
            }
            else if( value.isArray() )
            {
               ItemArray* arr = value.asArray();
               for( uint32 i = 0; i < arr->length(); ++i )
               {
                  Item& str = arr->at(i);

                  if( str.isString() )
                  {
                     URI::URLEncode( *str.asString(), sValue );
                     m_target += sName + "[]=" + sValue;
                  }
               }
            }

            // else, just drop the value
         }
      }

      String& m_target;
   };

   Rator rator(target);

   fields.enumerate(rator);
}


void dictAsInputFields( String& fwd, const ItemDict& items )
{
   class Rator: public ::Falcon::ItemDict::Enumerator
   {

   public:
      Rator( String& tgt ):
         m_target(tgt)
      {}

      virtual ~Rator() {}

      virtual void operator()( const Item& key, Item& value )
      {
         if ( key.isString() && value.isString() )
         {
            m_target += "<input type=\"hidden\" name=\"";
            htmlEscape( *key.asString(), m_target );
            m_target += "\" value=\"";
            htmlEscape( *value.asString(), m_target );
            m_target += "\"/>\n";
         }
      }

      String& m_target;
   };

   Rator rator(fwd);
   items.enumerate(rator);
}


void htmlEscape( const String& str, String& fwd )
{
   fwd.reserve(str.length());

   for ( Falcon::uint32 i = 0; i < str.length(); i++ )
   {
     Falcon::uint32 chr = str[i];
     switch ( chr )
     {
        case '<': fwd.append( "&lt;" ); break;
        case '>': fwd.append( "&gt;" ); break;
        case '"': fwd.append( "&quot;" ); break;
        case '&': fwd.append( "&amp;" ); break;
        default: fwd.append( chr ); break;
     }
   }
}


void parseQuery( const String &query, ItemDict& dict )
{
   // query is a set of zero or more & separated utf-8 url-encoded strings.

   // we must first find the & chars
   String section, key, value;
   uint32 pos1, pos2;

   pos1 = 0;
   pos2 = query.find( "&" );
   while ( pos1 != String::npos )
   {
      // get the substring
      section = query.subString( pos1, pos2 );
      parseQueryEntry( section, dict );

      // else, the record was malformed
      // What to do?

      if ( pos2 != String::npos )
      {
         pos1 = pos2 + 1;
         pos2 = query.find( "&", pos1 );
      }
      else
         break;
   }
}



void parseQueryEntry( const String &query, ItemDict& dict )
{
   bool proceed = false;
   String key;
   String value;

   // get the =
   uint32 poseq = query.find( "=" );
   if( poseq != String::npos )
   {
      if ( URI::URLDecode( query.subString( 0, poseq ), key ) &&
           URI::URLDecode( query.subString( poseq + 1 ), value )
         )
      {
         proceed = true;
      }

   }
   else
   {
      if( URI::URLDecode( query, key ) )
      {
         proceed = true;
         value = "";
      }
   }

   if ( proceed )
   {
      key.trim();
      value.bufferize();
      addQueryVariable( key, FALCON_GC_HANDLE(&(new String(value))->bufferize()), dict );
   }
}


void addQueryVariable( const String &key, const Item& value, ItemDict& dict )
{
   // is this a dictionary?
   if( ! key.endsWith("[]") )
   {
      dict.insert( FALCON_GC_HANDLE(&(new String(key))->bufferize()), value );
   }
   else
   {
      String short_key = key.subString(0, key.length()-2);
      // else, create an array with the given keys.
      Item *arr = dict.find( short_key );

      if ( arr != 0 )
      {
         if ( ! arr->isArray() )
         {
            Item* temp = arr;
            *arr = FALCON_GC_HANDLE(new ItemArray);
            arr->asArray()->append( *temp );
         }
         arr->asArray()->append( value );
      }
      else
      {
         ItemArray *carr = new ItemArray;
         carr->append( value );
         dict.insert( FALCON_GC_HANDLE(&(new String(short_key))->bufferize()), FALCON_GC_HANDLE(carr) );
      }
   }
}


bool parseHeaderEntry( const String &line, String& key, String& value )
{
   /*String l = sline;
   l.c_ize();
   String line;
   line.fromUTF8( (char*) l.getRawStorage() );*/

   uint32 pos = line.find( ":" );

   if( pos != String::npos )
   {
      key = line.subString( 0, pos );

      if( line.endsWith("\r\n") )
      {
         value = line.subString( pos+1, line.length()-2 );
      }
      else
      {
         value = line.subString( pos+1 );
      }

      unescapeQuotes( value );
      return true;
   }

   return false;
}


void makeRandomFilename( String& target, int size )
{
   static const char* alphabeth= "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
   target.reserve( size );

   MTRand_interlocked& rndgen = Engine::instance()->mtrand();
   for ( int i = 0; i < size; i ++ )
   {
      target.append( alphabeth[rndgen.randInt() % 63] );
   }
}

void unescapeQuotes( Falcon::String &str )
{
   str.trim();
   Falcon::uint32 len = str.length();
   if ( len > 1 )
   {
      if ( str.getCharAt(0) == '"' && str.getCharAt(len-1) == '"' )
      {
         str.remove(0,1);
         str.remove( len-2, 1 );
         len -= 2;
         Falcon::uint32 i = 0;
         while ( i + 1 < len )
         {
            if ( str.getCharAt(i) == '\\' &&  str.getCharAt(i+1) == '"' )
            {
               str.remove( i, 1 );
            }
            i++;
         }
      }
   }
}


void xrandomize()
{
   uint32 pid;

#ifdef FALCON_SYSTEM_WIN
   pid = (uint32) GetCurrentProcessId();
#else
   pid = (uint32) getpid();
#endif

   // time randomization is not enough
   // as we may be called in the same seconds from different processes.
   srand( (unsigned)time( NULL ) + pid );
}


}
}
}

/* end of utils.h */


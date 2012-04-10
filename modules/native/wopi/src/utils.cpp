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
#include <falcon/carray.h>
#include <falcon/coredict.h>
#include <falcon/cclass.h>
#include <falcon/lineardict.h>
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
   Iterator iter( const_cast<ItemDict*>(&fields) );

   while( iter.hasCurrent() )
   {
      if( target.size() != 0 )
        target += "&";

      String name, value;
      if( iter.getCurrentKey().isString() )
      {
         URI::URLEncode( *iter.getCurrentKey().asString(), name );

         if( iter.getCurrent().isString() )
         {
            URI::URLEncode( *iter.getCurrent().asString(), value );
            target += name + "=" + value;
         }
         else if( iter.getCurrent().isArray() )
         {
            CoreArray* arr = iter.getCurrent().asArray();
            for( uint32 i = 0; i < arr->length(); ++i )
            {
               Item& str = arr->at(i);

               if( str.isString() )
               {
                  URI::URLEncode( *str.asString(), value );
                  target += name + "[]=" + value;
               }
            }
         }

         // else, just drop the value
      }

      iter.next();
   }
}

void dictAsInputFields( String& fwd, const ItemDict& items )
{
   Falcon::Iterator iter( const_cast<ItemDict*>(&items) );
   while ( iter.hasCurrent() )
   {
      const Falcon::Item &key = iter.getCurrentKey();
      const Falcon::Item &value = iter.getCurrent();

      if ( key.isString() && value.isString() )
      {
         fwd += "<input type=\"hidden\" name=\"";
         htmlEscape( *key.asString(), fwd );
         fwd += "\" value=\"";
         htmlEscape( *value.asString(), fwd );
         fwd += "\"/>\n";
      }

      iter.next();
   }
}

void htmlEscape( const String& str, String& fwd )
{
   for ( Falcon::uint32 i = 0; i < str.length(); i++ )
   {
     Falcon::uint32 chr = str[i];
     switch ( chr )
     {
        case '<': fwd.append( "&lt;" ); break;
        case '>': fwd.append( "&gt;" ); break;
        case '"': fwd.append( "&quot;" ); break;
        case '&': fwd.append( "&amp;" ); break;
        default:
           fwd.append( chr );
     }
   }
}

CoreObject* makeURI( const URI& uri )
{
   VMachine* vm = VMachine::getCurrent();
   Item* i_uric = vm->findGlobalItem("URI");
   fassert( i_uric != 0 );
   fassert( i_uric->isClass() );

   return i_uric->asClass()->createInstance( new URI(uri), false );
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
   CoreString& value = *(new CoreString);

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

   key.trim();

   if ( proceed )
   {
      value.bufferize();
      addQueryVariable( key, &value, dict );
   }
}


void addQueryVariable( const String &key, const Item& value, ItemDict& dict )
{
   // is this a dictionary?
   if( key.endsWith("[]") )
   {
      String short_key = key.subString(0, key.length()-2);
      // create an array
      Item *arr = dict.find( short_key );

      if ( arr != 0 )
      {
         if ( ! arr->isArray() )
         {
            Item* temp = arr;
            *arr = new CoreArray;
            arr->asArray()->append( *temp );
         }
         arr->asArray()->append( value );
      }
      else
      {
         CoreArray *carr = new CoreArray;
         carr->append( value );
         dict.put( new CoreString( short_key ), carr );
      }
   }
   else if ( key.endsWith("]") )
   {
      // must be a dictionary entry.
      uint32 pos = key.find("[");
      if( pos == 0 || pos == String::npos )
      {
         // ignore and go on
         dict.put( new CoreString(key), value );
      }
      else 
      {
         String genPart = key.subString( 0, pos );
         String specPart = key.subString( pos+1, key.length()-1 );
         specPart.trim();
         if( (specPart[0] == '"' && specPart[specPart.length()-1] == '"') 
            || (specPart[0] == '\'' && specPart[specPart.length()-1] == '\'') )
         {
            specPart = specPart.subString(1,specPart.length()-1);
         }
         
         // else, create a dictionary
         Item *arr = dict.find( genPart );
         ItemDict* keyDict;
         if( arr == 0 )
         {
            keyDict = new LinearDict;
            dict.put( new CoreString( genPart), Item(new CoreDict(keyDict)) );
         }
         else 
         {
            if( ! arr->isDict() )
            {
               keyDict = new LinearDict;
               keyDict->put( Item(), *arr );
               dict.put( new CoreString(genPart), Item(new CoreDict(keyDict)) ); 
            }
            else 
            {
               keyDict = &arr->asDict()->items();
            }
         }
         
         keyDict->put( new CoreString(specPart), value );
      }
   }
   else
   {
      dict.put( new CoreString(key), value );
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

   for ( int i = 0; i < size; i ++ )
   {
      target.append( alphabeth[rand() % 63] );
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


/*
   FALCON - The Falcon Programming Language.
   FILE: strtable.cpp

   String table implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon Feb 14 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   String table implementation
*/

#include <falcon/strtable.h>
#include <falcon/common.h>
#include <falcon/memory.h>
#include <falcon/stream.h>
#include <falcon/string.h>
#include <falcon/traits.h>
#include <falcon/fassert.h>

namespace Falcon {

StringTable::StringTable():
   m_vector( &traits::t_stringptr_own() ),
   m_map( &traits::t_stringptr(), &traits::t_int() ),
   m_intMap( &traits::t_stringptr(), &traits::t_int() ),
   m_tableStorage(0),
   m_internatCount(0)
{}

StringTable::StringTable( const StringTable &other ):
   m_vector( &traits::t_stringptr_own() ),
   m_map( &traits::t_stringptr(), &traits::t_int() ),
   m_intMap( &traits::t_stringptr(), &traits::t_int() ),
   m_tableStorage(0),
   m_internatCount(0)
{
   for( uint32 i = 0; i < other.m_vector.size(); i ++ )
   {
      String* str = *(String**) other.m_vector.at(i);
      add( new String( *str ) );
   }
}


StringTable::~StringTable()
{
   if ( m_tableStorage != 0 )
      memFree( m_tableStorage );
}

int32 StringTable::add( String *str )
{
   fassert(str);
   
   if ( str->exported() )
   {
      if ( int32 *pos = (int32 *) m_intMap.find( str ) )
      {
         String *tableStr = (String *) m_vector.at( *pos );
         if (str != tableStr )
         {
           delete str;
           str = tableStr;
         }
         return *pos;
      }

      int32 id = m_vector.size();
      m_vector.push( str );
      m_intMap.insert( str, &id );
      m_internatCount++;
      return id;
   }
   else
   {
      if ( int32 *pos = (int32 *) m_map.find( str ) )
      {
         String *tableStr = (String *) m_vector.at( *pos );
         if ( str != tableStr )
         {
            delete str;
            str = tableStr;
         }
         return *pos;
      }

      int32 id = m_vector.size();
      m_vector.push( str );
      m_map.insert( str, &id );
      return id;
   }
}

String *StringTable::find( const String &source ) const
{
   MapIterator pos;
   if ( source.exported() )
   {
      if ( m_intMap.find( &source, pos ) )
         return *(String **) pos.currentKey();
   }
   else {
      if ( m_map.find( &source, pos ) )
         return *(String **) pos.currentKey();
   }
   return 0;
}

int32 StringTable::findId( const String &source ) const
{
   MapIterator pos;
   if ( source.exported() )
   {
      if ( m_intMap.find( &source, pos ) )
         return *(int32 *) pos.currentValue();
   }
   else {
      if ( m_map.find( &source, pos ) )
         return *(int32 *) pos.currentValue();
   }

   return -1;
}

bool StringTable::save( Stream *out ) const
{
   fassert(out); 
   
   // write a minimal table if no table is needed.
   uint32 f;
   if ( size() == 0 )
   {
      f = 0;
      out->write( &f, sizeof( f ) );
      return true;
   }

   f = endianInt32( size() );
   out->write( &f, sizeof( f ) );

   // serialize all the strings
   for( uint32 i = 0; i < m_vector.size(); i++ )
   {
      get(i)->serialize( out );
      if( ! out->good() )
         return false;
   }

   // align the stream to %4 == 0
   f = 0;
   while( out->good() && out->tell() %4 != 0 )
   {
      out->write( &f, 1 );
   }

   return true;
}

bool StringTable::load( Stream *in )
{
   fassert(in);
   
   int32 size;
   in->read( &size, sizeof( size ) );
   size = endianInt32( size );

   for( int i = 0; i < size; i ++ )
   {
      String *str = new String();
      // deserialize and create self-destroying static strings.
      if ( ! str->deserialize( in, false ) ) {
         delete str;
         return false;
      }

      // consider the strings in the string table static.
      // the destructor will destroy the data anyhow, as we have an allocated data
      add( str );
   }

   while( in->tell() %4 != 0 )
   {
      in->read( &size, 1 );
   }

   return true;
}

bool StringTable::skip( Stream *in ) const
{
   fassert(in);
  
   int32 size;
   in->read( &size, sizeof( size ) );
   size = endianInt32( size );

   for( int i = 0; i < size; i ++ )
   {
      String str;
      if ( ! str.deserialize( in ) ) {
         return false;
      }
   }

   while( in->tell() %4 != 0 )
   {
      in->read( &size, 1 );
   }

   return true;
}

bool StringTable::saveTemplate( Stream *out, const String &moduleName, const String &origLang ) const
{
   fassert(out);
   
   // we shouldn't even have been called if we have no international strings.
   if( m_internatCount == 0 )
      return true;

   // write the XML template
   // -- the caller should prepend the XML header if relevant
   String temp = "<translation module=\"" + moduleName + "\" from=\"" + origLang +
         "\" into=\"Your language code here\">\n";

   if ( ! out->writeString( temp ) )
      return false;

   for ( int i = 0; i < size(); i++ )
   {
      const String &current = *get(i);
      if( ! current.exported() )
         continue;

      //====== an international string.

      String temp = "<string id=\"";
      temp.writeNumber( (int64) i);
      temp+="\">\n<original>";

      if ( ! out->writeString( temp ) )
         return false;

      if ( ! out->writeString( current ) )
         return false;

      if ( ! out->writeString( "</original>\n<translated></translated>\n</string>\n" ) )
         return false;
   }

   if ( ! out->writeString( "</translation>\n" ) )
      return false;

   return true;
}

void StringTable::build( char **table, bool bInternational )
{
   fassert(table);
   char **ptr = table;
   while( *ptr != 0 )
   {
      String *s = new String( *ptr );
      s->exported( bInternational );
      add( s );
      ptr++;
   }
}

void StringTable::build( wchar_t **table, bool bInternational )
{
   fassert(table);
   wchar_t **ptr = table;
   while( *ptr != 0 )
   {
      String *s = new String( *ptr );
      s->exported( bInternational );
      add( s );
      ptr++;
   }
}

}


/* end of strtable.cpp */

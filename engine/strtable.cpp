/*
   FALCON - The Falcon Programming Language.
   FILE: strtable.cpp
   $Id: strtable.cpp,v 1.8 2007/03/18 19:21:13 jonnymind Exp $

   String table implementation
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon Feb 14 2005
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

namespace Falcon {

StringTable::StringTable():
   m_vector( &traits::t_stringptr_own ),
   m_map( &traits::t_stringptr, &traits::t_int ),
   m_tableStorage(0)
{}

StringTable::~StringTable()
{
   if ( m_tableStorage != 0 )
      memFree( m_tableStorage );
}

int32 StringTable::add( String *str )
{
   int32 *pos = (int32 *) m_map.find( str );

   if ( pos != 0 )
      return *pos;

   int32 id = m_vector.size();
   m_vector.push( str );
   str->id( id );
   m_map.insert( str, &id );
   return id;
}

String *StringTable::find( const String &source ) const
{
   MapIterator pos;
   if ( m_map.find( &source, pos ) )
      return *(String **) pos.currentKey();
   return 0;
}

int32 StringTable::findId( const String &str ) const
{
   MapIterator pos;
   if ( m_map.find( &str, pos ) )
      return *(int32 *) pos.currentValue();
   return -1;
}

bool StringTable::save( Stream *out ) const
{
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
   int32 size;
   in->read( &size, sizeof( size ) );
   size = endianInt32( size );

   for( int i = 0; i < size; i ++ )
   {
      String *str = new String();
      if ( ! str->deserialize( in ) ) {
         delete str;
         return false;
      }

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

bool StringTable::saveTemplate( Stream *out )
{
   for ( int i = 0; i < size(); i++ )
   {
      String temp;
      temp += "/* ";
      temp.writeNumber( (int64)i );
      temp +=" */\n";
      if ( ! out->writeString( temp ) )
         return false;

      if ( ! out->writeString( *get( i ) ) )
         return false;

      if ( ! out->writeString( "\n//=\n//==\n\n" ) )
         return false;
   }

   return true;
}

void StringTable::build( char **table )
{
   char **ptr = table;
   while( *ptr != 0 )
   {
      String *s = new String( *ptr );
      add( s );
      ptr++;
   }
}

void StringTable::build( wchar_t **table )
{
   wchar_t **ptr = table;
   while( *ptr != 0 )
   {
      add( new String( *ptr ) );
      ptr++;
   }
}

}


/* end of strtable.cpp */

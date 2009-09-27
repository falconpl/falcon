/*
   FALCON - The Falcon Programming Language.
   FILE: json_mod.cpp

   JSON transport format interface - inner logic (serviceable)
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 18:28:44 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/engine.h>

#include "json_mod.h"

namespace Falcon {

JSON::JSON( bool bEncUni, bool bPretty, bool bReadale ):
   m_bEncUnicode( bEncUni ),
   m_bPretty( bPretty ),
   m_bReadable( bReadale ),
   m_level(0)
{}

JSON::~JSON()
{}

bool JSON::encode( const Item& source, Stream* tgt  )
{
   String temp;
   for ( int i = 0; i < m_level; ++i )
      tgt->put( ' ' );

   switch ( source.type() )
   {
   case FLC_ITEM_NIL: tgt->writeString( "null" ); break;
   case FLC_ITEM_BOOL: tgt->writeString( source.asBoolean() ? "true" : "false" ); break;
   case FLC_ITEM_INT: tgt->writeString( temp.N( source.asInteger() ) ); break;
   case FLC_ITEM_NUM: tgt->writeString( temp.N( source.asNumeric() ) ); break;
   case FLC_ITEM_STRING:
      tgt->put( '"' );
      encode_string( *source.asString(), tgt );
      tgt->put( '"' );
      break;

   case FLC_ITEM_ARRAY:
      {
         if( source.asArray()->length() == 0 )
         {
            tgt->writeString( "[]" );
         }
         else
         {
            tgt->put( '[' );
            if( m_bReadable )
            {
               tgt->put( '\n' );
               m_level += 2;
            }
            else if ( m_bPretty )
               tgt->put( ' ' );

            const ItemArray& ci = source.asArray()->items();
            for( uint32 i = 0; i < ci.length(); ++i )
            {
               if( ! encode( ci[i], tgt ) )
                  return false;

               if( i + 1 < ci.length() )
               {
                  tgt->put( ',' );
                  if( m_bReadable )
                     tgt->put( '\n' );
                  else if ( m_bPretty )
                     tgt->put( ' ' );
               }
            }

            if( m_bReadable )
            {
               tgt->put( '\n' );
               m_level -= 2;
               for ( int i = 0; i < m_level; ++i )
                   tgt->put( ' ' );
            }
            else if ( m_bPretty )
               tgt->put( ' ' );

            tgt->put( ']' );
         }
      }
      break;

   case FLC_ITEM_DICT:
      {
         if( source.asDict()->length() == 0 )
         {
            tgt->writeString( "{}" );
         }
         else
         {
            tgt->put( '{' );
            if( m_bReadable )
            {
               tgt->put( '\n' );
               m_level += 2;
            }
            else if ( m_bPretty )
               tgt->put( ' ' );


            Iterator iter( &source.asDict()->items() );
            while( iter.hasCurrent() )
            {
               String name;
               iter.getCurrentKey().toString( name );
               tgt->put( '"' );
               encode_string(name, tgt);
               tgt->put( '"' );
               tgt->put( ':' );
               if ( m_bPretty || m_bReadable )
                  tgt->put( ' ' );

               if( ! encode( iter.getCurrent(), tgt ) )
                  return false;

               iter.next();
               if( iter.hasCurrent() )
               {
                  tgt->put( ',' );
                  if( m_bReadable )
                     tgt->put( '\n' );
                  else if ( m_bPretty )
                     tgt->put( ' ' );
               }

            }

            if( m_bReadable )
            {
               tgt->put( '\n' );
               m_level -= 2;
               for ( int i = 0; i < m_level; ++i )
                   tgt->put( ' ' );
            }
            else if ( m_bPretty )
               tgt->put( ' ' );

            tgt->put( '}' );
         }
      }
      break;

   case FLC_ITEM_OBJECT:
      {
         const CoreObject* obj = source.asObjectSafe();
         const PropertyTable& tab = obj->generator()->properties();

         tgt->put( '{' );
         if( m_bReadable )
         {
            tgt->put( '\n' );
            m_level += 2;
         }
         else if ( m_bPretty )
            tgt->put( ' ' );

         for ( uint32 i = 0; i < tab.added(); ++i )
         {
            Item prop;
            if ( obj->getProperty( *tab.getKey(i), prop ) &&
                  prop.isNil() || prop.isOrdinal() || prop.isBoolean()
                  || prop.isString() || prop.isDict() || prop.isArray() )
            {
               tgt->put( '"' );
               encode_string( *tab.getKey( i ), tgt);
               tgt->put( '"' );

               tgt->put( ':' );
               if ( m_bPretty || m_bReadable )
                  tgt->put( ' ' );

               if( ! encode( prop, tgt ) )
                  return false;
            }

            if( i+1 < tab.added() )
            {
               tgt->put( ',' );
               if( m_bReadable )
                  tgt->put( '\n' );
               else if ( m_bPretty )
                  tgt->put( ' ' );
            }

         }

         if( m_bReadable )
         {
            tgt->put( '\n' );
            m_level -= 2;
            for ( int i = 0; i < m_level; ++i )
                tgt->put( ' ' );
         }
         else if ( m_bPretty )
            tgt->put( ' ' );

         tgt->put( '}' );
      }
      break;


   default: return false;
   }

   return true;
}


void JSON::encode_string( const String &str, Stream* tgt ) const
{
   uint32 len = str.length();
   uint32 pos = 0;

   while( pos < len )
   {
      uint32 chat = str.getCharAt( pos );
      switch( chat )
      {
         case '"': tgt->writeString( "\\\"" ); break;
         case '\r': tgt->writeString( "\\r" ); break;
         case '\n': tgt->writeString( "\\n" ); break;
         case '\t': tgt->writeString( "\\t" ); break;
         case '\f': tgt->writeString( "\\f" ); break;
         case '\b': tgt->writeString( "\\b" ); break;
         case '\\': tgt->writeString( "\\\\" ); break;
         default:
            if ( chat < 8 || ( m_bEncUnicode && chat >127 )) {
               char bufarea[14];
               bufarea[0] = '\\';
               bufarea[1] = 'u';

               if( chat > 0xFFFF )
                  chat = 0xFFFF;

               String::uint32ToHex( chat, bufarea+2 );
               tgt->writeString( bufarea );
            }
            else{
               tgt->put( chat );
            }
      }
      pos++;
   }

}

bool JSON::decode( Item& target, Stream* src ) const
{

}

}

/* end of json_mod.cpp */

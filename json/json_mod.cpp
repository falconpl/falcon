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
                  (prop.isNil() || prop.isOrdinal() || prop.isBoolean()
                  || prop.isString() || prop.isDict() || prop.isArray() ))
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
   String temp;

   enum {
      st_none,
      st_number,
      st_float,
      st_firstexp,
      st_exp,
      st_string,
      st_escape
   } state;

   state = st_none;
   uint32 chr;
   while( src->get( chr ) )
   {
      switch( state )
      {
         case st_none:
            switch( chr )
            {
               // ignore ws
               case ' ': case '\t': case '\r': case '\n': continue;
               case '+': state = st_number; break;
               case '-': state = st_number; temp.append(chr); break;
               case '"': state = st_string; break;
               case '0': case '1': case '2': case '3': case '4':
               case '5': case '6': case '7': case '8': case '9':
                  state = st_number; temp.append(chr);
                  break;

               // try to read true, false or null
               case 'n':
                  if ( ! src->get( chr ) || chr != 'u' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( src->get( chr ) &&
                       chr != ',' && chr != ' ' && chr != '\t' && chr != '\r' && chr != '\n'
                       )
                     return false;

                  target.setNil();
                  return true;

               case 't':
                  if ( ! src->get( chr ) || chr != 'r' ) return false;
                  if ( ! src->get( chr ) || chr != 'u' ) return false;
                  if ( ! src->get( chr ) || chr != 'e' ) return false;
                  if ( src->get( chr ) &&
                       chr != ',' && chr != ' ' && chr != '\t' && chr != '\r' && chr != '\n'
                       )
                     return false;

                  target.setBoolean( true );
                  return true;

               case 'f':
                  if ( ! src->get( chr ) || chr != 'a' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( ! src->get( chr ) || chr != 's' ) return false;
                  if ( ! src->get( chr ) || chr != 'e' ) return false;
                  if ( src->get( chr ) &&
                       chr != ',' && chr != ' ' && chr != '\t' && chr != '\r' && chr != '\n'
                       )
                     return false;

                  target.setBoolean( false );
                  return true;

               case '{':
               {
                  CoreDict* cd = decodeDict( src );
                  if ( cd == 0 )
                     return false;
                  target = cd;
               }
               return true;

               case '[':
               {
                  CoreArray* ca = decodeArray( src );
                  if ( ca == 0 )
                     return false;
                  target = ca;
               }
               return true;

               default:
                  return false;
            }
            break;

         case st_number:  case st_float:
            if( chr >= '0' || chr <= '9' )
            {
               temp.append(chr);
            }
            else if( chr == '.' )
            {
               if ( state == st_float )
                  return false;

               state = st_float;
               temp.append(chr);
            }
            else if( chr == 'e' || chr == 'E' )
            {
               state = st_firstexp;
               temp.append(chr);
            }
            else
            {
               src->unget( chr );
               if ( state == st_float )
               {
                  numeric number;
                  temp.parseDouble( number );
                  target = number;
               }
               else
               {
                  int64 number;
                  temp.parseInt( number );
                  target = number;
               }
               return true;
            }

         break;

         case st_firstexp:
            if( chr >= '0' || chr <= '9' || chr == '-' )
            {
               temp.append(chr);
               state = st_exp;
            }
            else
               return false;
         break;

         case st_exp:
            if( chr >= '0' || chr <= '9' || chr == '-' )
            {
               temp.append(chr);
               state = st_exp;
            }
            else
            {
               src->unget( chr );
               numeric number;
               temp.parseDouble( number );
               target = number;
               return true;
            }
            return false;

         break;

         case st_string:
            if ( chr == '"' )
            {
               target = new CoreString( temp );
               return true;
            }
            else if ( chr == '\\' )
            {
               state = st_escape;
            }
            else
            {
               temp.append( chr );
            }
            break;

         case st_escape:
            switch( chr )
            {
               case '\\': temp.append( '\\' ); break;
               case 'b': temp.append( '\b' ); break;
               case 't': temp.append( '\t' ); break;
               case 'n': temp.append( '\n' ); break;
               case 'r': temp.append( '\r' ); break;
               case 'f': temp.append( '\f' ); break;
               case '/': temp.append( '/' ); break;
               default:
                  return false;
            }
            break;

      }
   }

   return false;
}


CoreArray* JSON::decodeArray( Stream* src ) const
{
   CoreArray* ca = new CoreArray;
   uint32 chr;

   bool bComma = false;
   while( src->get( chr ) )
   {
      if( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
         continue;

      if ( chr == ']' )
         return ca;

      if ( bComma )
      {
         if ( chr == ',' )
            bComma = false;
         else
            return 0;
      }
      else
      {
         Item tmp;
         src->unget( chr );
         if( ! decode( tmp, src ) )
         {
            // dispose all the data.
            ca->gcMark(0);
            return 0;
         }
         ca->append( tmp );
         bComma = true;
      }
   }

   // decoding is incomplete
   // dispose all the data.
   ca->gcMark(0);
   return 0;

}

CoreDict* JSON::decodeDict( Stream* src ) const
{
   LinearDict* cd = new LinearDict;
   uint32 chr;

   enum {
      st_key,
      st_value,
      st_comma,
      st_colon
   } state;

   state = st_key;

   Item key, value;
   bool bComma = false;
   while( src->get( chr ) )
   {
      if( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
         continue;

      if ( state == st_colon )
      {
         if ( chr == ':' )
            state = st_value;
         else
         {
            cd->gcMark(0);
            delete cd;
            return 0;
         }

      }
      else if ( state == st_comma )
      {
         if ( chr == '}' )
            return new CoreDict( cd );

         if ( chr == ',' )
            bComma = false;
         else
         {
            cd->gcMark(0);
            delete cd;
            return 0;
         }
      }
      else if ( state == st_key )
      {
         if( ! decode( key, src ) || key.isString() )
         {
            // dispose all the data.
            cd->gcMark(0);
            delete cd;
            return 0;
         }
         state = st_colon;
      }
      else
      {
         if( ! decode( value, src ) )
         {
            // dispose all the data.
            cd->gcMark(0);
            delete cd;
            return 0;
         }
         state = st_key;
         cd->put( key, value );
      }

   }

   // decoding is incomplete
   // dispose all the data.
   cd->gcMark(0);
   delete cd;
   return 0;

}

}

/* end of json_mod.cpp */

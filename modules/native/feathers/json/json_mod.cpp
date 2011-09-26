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

JSON::JSON( bool bPretty, bool bReadale ):
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

               if( i+1 < tab.added() )
               {
                  tgt->put( ',' );
                  if( m_bReadable )
                     tgt->put( '\n' );
                  else if ( m_bPretty )
                     tgt->put( ' ' );
               }
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

inline bool isterminal( char chr )
{
	return
		chr == ','
		|| chr == '}'
		|| chr == ']'
		|| chr == ' '
		|| chr == '\t'
		|| chr == '\r'
		|| chr == '\n';
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
            if ( chat < 32 || chat >127 ) {
               String temp = "\\u";
               temp.H( chat, true, 4 );
               tgt->writeString( temp );
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
      st_exp
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
               case '"': case '\'':
                  {
                     String str;
                     src->unget(chr);
                     if ( ! decodeKey( str, src ) )
                        return false;
                     target = new CoreString( str );
                     return true;
                  }
                  break;

               case '0': case '1': case '2': case '3': case '4':
               case '5': case '6': case '7': case '8': case '9':
                  state = st_number; temp.append(chr);
                  break;

               // try to read true, false or null
               case 'n':
                  if ( ! src->get( chr ) || chr != 'u' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( ! src->get( chr ) || ! isterminal( chr ) )
                     return false;

                  target.setNil();
                  src->unget( chr );
                  return true;

               case 't':
                  if ( ! src->get( chr ) || chr != 'r' ) return false;
                  if ( ! src->get( chr ) || chr != 'u' ) return false;
                  if ( ! src->get( chr ) || chr != 'e' ) return false;
                  if ( ! src->get( chr ) || ! isterminal( chr ) )
                        return false;

                  target.setBoolean( true );
                  src->unget( chr );
                  return true;

               case 'f':
                  if ( ! src->get( chr ) || chr != 'a' ) return false;
                  if ( ! src->get( chr ) || chr != 'l' ) return false;
                  if ( ! src->get( chr ) || chr != 's' ) return false;
                  if ( ! src->get( chr ) || chr != 'e' ) return false;
                  if ( ! src->get( chr ) || ! isterminal( chr ) )
                        return false;

                  target.setBoolean( false );
                  src->unget( chr );
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
            if( chr >= '0' && chr <= '9' )
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
            if( (chr >= '0' && chr <= '9') || chr == '-' )
            {
               temp.append(chr);
               state = st_exp;
            }
            else
               return false;
         break;

         case st_exp:
            if( (chr >= '0' && chr <= '9') || chr == '-' )
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
         else if ( chr == ']' )
            return ca;
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
      st_key_first,
      st_key,
      st_value,
      st_comma,
      st_colon
   } state;

   state = st_key_first;

   Item key, value;

   while( src->get( chr ) )
   {
      if( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
         continue;

      switch( state )
      {
      case st_colon:
         if ( chr == ':' )
            state = st_value;
         else
         {
            cd->gcMark(0);
            delete cd;
            return 0;
         }
         break;

      case st_comma:
         if ( chr == '}' )
            return new CoreDict( cd );

         if ( chr == ',' )
            state = st_key;
         else
         {
            cd->gcMark(0);
            delete cd;
            return 0;
         }
         break;

         case st_key_first:
            // empty dict?
            if( chr == '}' )
            {
               return new CoreDict( cd );
            }
            // else, fallthrough
            
         case st_key: 
         src->unget( chr );
         {
            String sKey;
            if( ! decodeKey( sKey, src ) || sKey.size() == 0 )
            {
               // dispose all the data.
               cd->gcMark(0);
               delete cd;
               return 0;
            }
            key = new CoreString( sKey );
         }

         state = st_colon;
         break;

      case st_value:
         src->unget( chr );
         if( ! decode( value, src ) )
         {
            // dispose all the data.
            cd->gcMark(0);
            delete cd;
            return 0;
         }

         state = st_comma;
         cd->put( key, value );
         break;
      }
   }

   // decoding is incomplete
   // dispose all the data.
   cd->gcMark(0);
   delete cd;
   return 0;

}


bool JSON::decodeKey( String& tgt, Stream* src ) const
{
   // a key may be a symbol or a string

   uint32 chr;
   if ( ! src->get( chr ) )
      return false;

   if( chr == '"' || chr == '\'')
   {
      uint32 ch1 = chr;
      uint32 unicount = 4;
      uint32 unival = 0;

      while( src->get( chr ) && chr != ch1 )
      {
         if ( chr == '\\' )
         {
            src->get(chr);
            switch( chr )
            {
               case '\\': tgt.append( '\\' ); break;
               case '"': tgt.append( '"' ); break;
               case 'b': tgt.append( '\b' ); break;
               case 't': tgt.append( '\t' ); break;
               case 'n': tgt.append( '\n' ); break;
               case 'r': tgt.append( '\r' ); break;
               case 'f': tgt.append( '\f' ); break;
               case '/': tgt.append( '/' ); break;
               case 'u': unival = 0; unicount = 0; break;
               default:
                  return false;
            }
         }
         else if (unicount < 4 )
         {
            if ( chr >= '0' && chr <= '9' )
            {
               unival <<= 4;
               unival |= (unsigned char) (chr - '0');
            }
            else if ( chr >= 'a' && chr <= 'f' )
            {
               unival <<= 4;
               unival |= (unsigned char) (chr - 'a')+10;
            }
            else if ( chr >= 'A' && chr <= 'F' )
            {
               unival <<= 4;
               unival |= (unsigned char) (chr - 'A')+10;
            }
            else
               return false;

            if ( ++unicount == 4 )
            {
               tgt.append( unival );
            }
         }
         else
            tgt.append( chr );
      }

      // Eof?
      if ( chr != ch1 )
         return false;
   }
   else
   {
      tgt.append( chr );

      while( src->get( chr ) )
      {
         if ( chr == '\n' || chr == '\r' || chr == '\t' || chr == ' ' || chr == ',' || chr == ':')
         {
            src->unget(chr);
            return true;
         }
         tgt.append( chr );
      }
   }

   // this happens at eof; we return true, but other things may return false
   return true;
}

}

/* end of json_mod.cpp */

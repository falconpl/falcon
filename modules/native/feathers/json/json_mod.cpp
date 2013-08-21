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

#include <falcon/stream.h>
#include <falcon/item.h>
#include <falcon/textwriter.h>
#include <falcon/textreader.h>
#include <falcon/itemarray.h>
#include <falcon/itemdict.h>
#include <falcon/stdhandlers.h>

#include "json_mod.h"

namespace Falcon {

JSON::JSON( Stream* stream, bool bPretty, bool bReadale ):
   m_bPretty( bPretty ),
   m_bReadable( bReadale ),
   m_level(0),
   m_stream( stream )
{
   m_stream->incref();
   m_tw = new TextWriter( m_stream );
   m_tr = new TextReader( m_stream );
   m_charPos = 0;
   m_oldCharPos = 0;
   m_linePos = 0;
}

JSON::~JSON()
{
   m_stream->decref();
   m_tr->decref();
   m_tw->decref();
}

bool JSON::encode( const Item& source, String& error )
{
   String temp;
   for ( int i = 0; i < m_level; ++i )
      m_tw->putChar( ' ' );

   switch ( source.type() )
   {
   case FLC_ITEM_NIL: m_tw->write( "null" ); break;
   case FLC_ITEM_BOOL: m_tw->write( source.asBoolean() ? "true" : "false" ); break;
   case FLC_ITEM_INT: m_tw->write( temp.N( source.asInteger() ) ); break;
   case FLC_ITEM_NUM: m_tw->write( temp.N( source.asNumeric() ) ); break;
   case FLC_CLASS_ID_STRING:
      m_tw->putChar( '"' );
      encode_string( *source.asString() );
      m_tw->putChar( '"' );
      break;

   case FLC_CLASS_ID_ARRAY:
      {
         if( source.asArray()->length() == 0 )
         {
            m_tw->write( "[]" );
         }
         else
         {
            m_tw->putChar( '[' );
            if( m_bReadable )
            {
               m_tw->putChar( '\n' );
               m_level += 2;
            }
            else if ( m_bPretty )
            {
               m_tw->putChar( ' ' );
            }

            const ItemArray& ci = *source.asArray();
            for( uint32 i = 0; i < ci.length(); ++i )
            {
               if( ! encode( ci[i], error ) )
               {
                  return false;
               }

               if( i + 1 < ci.length() )
               {
                  m_tw->putChar( ',' );
                  if( m_bReadable )
                  {
                     m_tw->putChar( '\n' );
                  }
                  else if ( m_bPretty )
                  {
                     m_tw->putChar( ' ' );
                  }
               }
            }

            if( m_bReadable )
            {
               m_tw->putChar( '\n' );
               m_level -= 2;
               for ( int i = 0; i < m_level; ++i )
               {
                  m_tw->putChar( ' ' );
               }
            }
            else if ( m_bPretty )
            {
               m_tw->putChar( ' ' );
            }

            m_tw->putChar( ']' );
         }
      }
      break;

   case FLC_CLASS_ID_DICT:
      {
         if( source.asDict()->empty() )
         {
            m_tw->write( "{}" );
         }
         else
         {
            m_tw->putChar( '{' );
            if( m_bReadable )
            {
               m_tw->putChar( '\n' );
               m_level += 2;
            }
            else if ( m_bPretty )
            {
               m_tw->putChar( ' ' );
            }

            ItemDict* dict = source.asDict();
            class Rator: public ItemDict::Enumerator
            {
            public:
               Rator( JSON* js ): m_js(js),  m_bFirst(true) {}
               virtual ~Rator() {}
               virtual void operator()( const Item& key, Item& value )
               {
                  if( ! key.isString() )
                  {
                     Class* cls = 0;
                     void* udata;
                     key.forceClassInst(cls, udata);
                     throw(String("Dictionary has key entry of class \"") + cls->name() + "\"");
                  }

                  if( m_bFirst )
                  {
                     m_bFirst = false;
                  }
                  else {
                     m_js->putLine( ',' );
                  }

                  String error;
                  if( ! m_js->encode( key, error ) )
                  {
                     throw error;
                  }

                  m_js->put( ':' );
                  if( ! m_js->encode( value, error ) )
                  {
                     throw error;
                  }
               }

            private:
               JSON* m_js;
               bool m_bFirst;
            }
            rator( this );

            try {
               dict->enumerate(rator);
            }
            catch( String& err )
            {
               error = err;
               return false;
            }

            if( m_bReadable )
            {
               m_tw->putChar( '\n' );
               m_level -= 2;
               for ( int i = 0; i < m_level; ++i )
               {
                  m_tw->putChar( ' ' );
               }
            }
            else if ( m_bPretty )
            {
               m_tw->putChar( ' ' );
            }

            m_tw->putChar( '}' );
         }
      }
      break;


   default: return false;
   }

   m_tw->flush();
   return true;
}

void JSON::put( uint32 chr )
{
   if( m_tw != 0 )
   {
      m_tw->putChar( chr );
      if ( m_bPretty || m_bReadable )
      {
         m_tw->putChar( ' ' );
      }
   }
}

void JSON::putLine( uint32 chr )
{
   if( m_tw != 0 )
   {
      m_tw->putChar( chr );
      if( m_bReadable )
         m_tw->putChar( '\n' );
      else if ( m_bPretty )
         m_tw->putChar( ' ' );
   }
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

void JSON::encode_string( const String &str ) const
{
   uint32 len = str.length();
   uint32 pos = 0;

   while( pos < len )
   {
      uint32 chat = str.getCharAt( pos );
      switch( chat )
      {
         case '"': m_tw->write( "\\\"" ); break;
         case '\r': m_tw->write( "\\r" ); break;
         case '\n': m_tw->write( "\\n" ); break;
         case '\t': m_tw->write( "\\t" ); break;
         case '\f': m_tw->write( "\\f" ); break;
         case '\b': m_tw->write( "\\b" ); break;
         case '\\': m_tw->write( "\\\\" ); break;
         default:
            if ( chat < 32 || chat >127 ) {
               String temp = "\\u";
               temp.H( chat, true, 4 );
               m_tw->write( temp );
            }
            else{
               m_tw->putChar( chat );
            }
            break;
      }
      pos++;
   }

}

bool JSON::getChar( uint32& chr )
{
   if( m_tr->eof() )
   {
      return false;
   }

   chr = m_tr->getChar();
   if( chr == '\n' )
   {
      m_oldCharPos = m_charPos;
      m_charPos = 0;
      m_linePos++;
   }
   else {
      m_charPos++;
   }

   return true;
}

void JSON::ungetChar( uint32 chr )
{
   if( chr == '\n' ) {
      m_charPos = m_oldCharPos;
      m_linePos --;
   }
   else {
      m_charPos--;
   }

   m_tr->ungetChar(chr);
}

void JSON::setError( const String& desc, String& target ) const
{
   target = desc;
   target += " at ";
   target.N(m_linePos);
   target.A(":");
   target.N(m_charPos);
}

bool JSON::decode( Item& target, String& error )
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
   while( getChar(chr) )
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
                     ungetChar(chr);

                     if ( ! decodeKey( str ) )
                     {
                        setError( "Failed decoding of string", error );
                        return false;
                     }
                     target = FALCON_GC_HANDLE( new String(str) );
                     return true;
                  }
                  break;

               case '0': case '1': case '2': case '3': case '4':
               case '5': case '6': case '7': case '8': case '9':
                  state = st_number; temp.append(chr);
                  break;

               // try to read true, false or null
               case 'n':
                  if ( ! getChar( chr ) || chr != 'u' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'l' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'l' ) goto invalid;
                  if ( ! getChar( chr ) || ! isterminal( chr ) ) goto invalid;

                  target.setNil();
                  ungetChar(chr);
                  return true;

               case 't':
                  if ( ! getChar( chr ) || chr != 'r' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'u' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'e' ) goto invalid;
                  if ( ! getChar( chr ) || ! isterminal( chr ) ) goto invalid;

                  target.setBoolean( true );
                  ungetChar(chr);
                  return true;

               case 'f':
                  if ( ! getChar( chr ) || chr != 'a' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'l' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 's' ) goto invalid;
                  if ( ! getChar( chr ) || chr != 'e' ) goto invalid;
                  if ( ! getChar( chr ) || ! isterminal( chr ) ) return false;

                  target.setBoolean( false );
                  ungetChar( chr );
                  return true;

               case '{':
               {
                  if ( ! decodeDict( target, error ) )
                     return false;
               }
               return true;

               case '[':
               {
                  if ( ! decodeArray( target, error ) )
                     return false;
               }
               return true;

               default:
                  setError( "Parsing failed", error );
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
               ungetChar( chr );
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
            {
               setError("Invalid number format", error);
               return false;
            }
         break;

         case st_exp:
            if( (chr >= '0' && chr <= '9') || chr == '-' )
            {
               temp.append(chr);
               state = st_exp;
            }
            else
            {
               ungetChar( chr );
               numeric number;
               temp.parseDouble( number );
               target = number;
               return true;
            }
            setError("Invalid number format", error);
            return false;

         break;

      }
   }

invalid:
   setError( "Invalid value", error );
   return false;
}


bool JSON::decodeArray( Item& target, String& error )
{
   ItemArray* ca = new ItemArray;
   target = FALCON_GC_HANDLE(ca);
   uint32 chr;

   bool bComma = false;
   while( getChar( chr ) )
   {
      if( chr == ' ' || chr == '\t' || chr == '\r' || chr == '\n' )
         continue;

      if ( chr == ']' )
         return true;

      if ( bComma )
      {
         if ( chr == ',' )
            bComma = false;
         else if ( chr == ']' )
            return true;
         else
         {
            setError( "Invalid array format", error );
            return false;
         }
      }
      else
      {
         Item tmp;
         ungetChar( chr );
         if( ! decode( tmp, error ) )
         {
            return false;
         }
         ca->append( tmp );
         bComma = true;
      }
   }

   // decoding is incomplete
   // dispose all the data.
   setError("Hit EOF while decoding array", error);
   return false;
}

bool JSON::decodeDict( Item& target, String& error )
{
   ItemDict* cd = new ItemDict;
   target = FALCON_GC_HANDLE(cd);
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

   while( getChar( chr ) )
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
            setError( "Invalid dictionary format", error );
            return false;
         }
         break;

      case st_comma:
         if ( chr == '}' )
            return true;

         if ( chr == ',' )
            state = st_key;
         else
         {
            setError( "Invalid dictionary format", error );
            return false;
         }
         break;

         case st_key_first:
            // empty dict?
            if( chr == '}' )
            {
               return true;
            }
            // else, fallthrough
            /* no break */
            
         case st_key: 
         ungetChar( chr );
         {
            String sKey;
            if( ! decodeKey( sKey ) || sKey.size() == 0 )
            {
               setError( "Invalid key in dictionary", error );
               return false;
            }
            key = FALCON_GC_HANDLE(new String( sKey ));
         }

         state = st_colon;
         break;

      case st_value:
         ungetChar( chr );
         if( ! decode( value, error ) )
         {
            return false;
         }

         state = st_comma;
         cd->insert( key, value );
         break;
      }
   }

   // decoding is incomplete
   setError("Hit EOF while decoding dictionary", error);
   return false;
}


bool JSON::decodeKey( String& tgt )
{
   // a key may be a symbol or a string

   uint32 chr;
   if ( m_tr->eof() )
      return false;

   chr = m_tr->getChar();
   if( chr == '"' || chr == '\'')
   {
      uint32 ch1 = chr;
      uint32 unicount = 4;
      uint32 unival = 0;

      while( getChar( chr ) && chr != ch1 )
      {
         if ( chr == '\\' )
         {
            getChar(chr);
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

      while( getChar( chr ) )
      {
         if ( chr == '\n' || chr == '\r' || chr == '\t' || chr == ' ' || chr == ',' || chr == ':')
         {
            ungetChar(chr);
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

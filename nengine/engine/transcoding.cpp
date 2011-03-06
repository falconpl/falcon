/*
   FALCON - The Falcon Programming Language
   FILE: transcoding.cpp

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer ago 23 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Short description
*/

#include <falcon/memory.h>
#include <falcon/transcoding.h>
#include <falcon/stream.h>
#include <falcon/stringstream.h>
#include <falcon/sys.h>

#include <falcon/stdstreams.h>

#include <string.h>
#include <locale.h>
#include <stdlib.h>
#include <stdio.h>

namespace Falcon {

#include "trans_tables.h"



/** UTF-16LE encoding transcoder. */
class TranscoderUTF16LE: public TranscoderUTF16
{
public:
   TranscoderUTF16LE( Stream *s, bool bOwn=false ):
     TranscoderUTF16( s, bOwn, e_le )
   {
      m_bom = false;
   }

     virtual const String encoding() const { return "utf-16LE"; }
};

/** UTF-16BE encoding transcoder. */
class TranscoderUTF16BE: public TranscoderUTF16
{
public:
   TranscoderUTF16BE( Stream *s, bool bOwn=false ):
     TranscoderUTF16( s, bOwn, e_be )
   {
      m_bom = false;
   }

   virtual const String encoding() const { return "utf-16BE"; }
};

/** GBK encoding transcoder. */
class TranscoderGBK: public Transcoder
{
private:
   Table *decoderTable1;
   Table *decoderTable2;
   Table *encoderTable1;
   Table *encoderTable2;
   const static uint32  start = 0x40;
   const static uint32  end =  0xFE;

public:
   TranscoderGBK( const TranscoderGBK &other ):
      Transcoder( other ),
         decoderTable1( other.decoderTable1 ),
         decoderTable2( other.decoderTable2 ),
         encoderTable1( other.encoderTable1 ),
         encoderTable2( other.encoderTable2 )
      {}

public:
   TranscoderGBK( Stream *s, bool bOwn=false ):
      Transcoder(s, bOwn)
      {
         decoderTable1 = &gbkDecoderTable1;
         decoderTable2 = &gbkDecoderTable2;
         encoderTable1 = &gbkEncoderTable1;
         encoderTable2 = &gbkEncoderTable2;
      }

private:
   inline uint16 getUint16(Table *t, uint32 pos)
   {
      return ( (uint16*)t->table )[pos];
   }

   inline Table *getTable(Table *t, uint32 pos)
   {
      return ( (Table*)t->table ) + pos;
   }

   uint32 decodeDouble(uint32 byte1, uint32 byte2) {
      if ( (byte2 < start) || (byte2 > end) )
         return REPLACE_CHAR;

      int n0 = getUint16(decoderTable1, byte1) & 0xf;
      int n = n0 * (end - start + 1) + (byte2 - start);
      int ti = getUint16(decoderTable1, byte1)>>4; // Table 2 Index
      Table *charTable = getTable(decoderTable2, ti);
      return getUint16(charTable, n);
   }

   uint32 encodeDouble(uint32 ch) {
      int offset = getUint16(encoderTable1, ((ch & 0xff00) >> 8 )) << 8;
      Table *charTable = getTable(encoderTable2, offset >> 12);
      return getUint16(charTable, (offset & 0xfff) + (ch & 0xff));
   }

public:
   virtual const String encoding() const { return "gbk"; }

   virtual TranscoderGBK *clone() const
   {
      return new TranscoderGBK( *this );
   }

public:
   virtual bool get( uint32 &chr )
   {
      m_parseStatus = true;

      if( popBuffer( chr ) )
         return true;

      // converting the byte array into an unicode.
      byte b1;
      byte b2=0;
      if ( m_stream->read( &b1, 1 ) != 1 )
         return false;

      if (b1 < 0x80)
      {
         chr = b1;
      }
      else if ( b1&0x7F )
      {
         if ( b1^0xff )
         {
            if (m_stream->read( &b2, 1 ) != 1)
               return false;
            chr = decodeDouble(b1, b2);
         }
         else 
         {
            chr = 0xf8f5;
         }
      }
      else 
      {
         chr = 0x20ac;
      }

      if (chr == REPLACE_CHAR)
      {
         m_parseStatus = false;
      }

      return true;
   }

   virtual bool put( uint32 chr )
   {
      m_parseStatus = true;

      byte bPtr[2];
      if ( chr < 0x80 )
      {
         bPtr[0] = (byte)chr;
         return ( m_stream->write( bPtr, 1 ) == 1 );
      }


      uint32 ncode = encodeDouble( chr );
      if ( ncode == 0 || chr == 0 )
      {
         m_parseStatus = false;
         return true;
      }

      bPtr[0] = (byte)( (ncode & 0xff00) >> 8 );

      bPtr[1] = (byte)( ncode );
      if (bPtr[0]>0)
         return ( m_stream->write(bPtr, 2) == 2 );
      else
         return ( m_stream->write(bPtr + 1, 1) == 1 );
   }
};

/** Base class for codepage like transcoders. */
class TranscoderISO_CP: public Transcoder
{
protected:
   TranscoderISO_CP( Stream *s, bool bOwn=false ):
       Transcoder( s, bOwn )
       {
          m_nMinUntl = 0x0;
          m_nMaxUntl = 0x9F;
       }

    TranscoderISO_CP( const TranscoderISO_CP &other );

       uint16 *m_directTable;
       uint32 m_dirTabSize;
       CP_ISO_UINT_TABLE *m_reverseTable;
       uint32 m_revTabSize;
       uint32 m_nMinUntl;
       uint32 m_nMaxUntl;

public:
   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
};


class TranscoderOEM: public TranscoderISO_CP
{
protected:
   TranscoderOEM( Stream *s, bool bOwn=false ):
       TranscoderISO_CP( s, bOwn )
       {
          m_nMinUntl = 0x00;
          m_nMaxUntl = 0x7F;
       }

    TranscoderOEM( const TranscoderOEM &other );
public:
   virtual bool get( uint32 &chr );
};


class TranscoderIBM437:public TranscoderOEM
{
public:
   TranscoderIBM437( const TranscoderIBM437 &other ):
     TranscoderOEM( other )
     {}

     TranscoderIBM437( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "IBM437"; }
     virtual TranscoderIBM437 *clone() const;
};

class TranscoderCP1252:public TranscoderISO_CP
{
public:
   TranscoderCP1252( const TranscoderCP1252 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderCP1252( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "cp1252"; }
     virtual TranscoderCP1252 *clone() const;
};

class TranscoderIBM850:public TranscoderOEM
{
public:
   TranscoderIBM850( const TranscoderIBM850 &other ):
     TranscoderOEM( other )
     {}

     TranscoderIBM850( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "IBM850"; }
     virtual TranscoderIBM850 *clone() const;
};


/** Latin-1 (ISO8859_1) transcoder. */
class TranscoderISO8859_1: public TranscoderISO_CP
{
public:
   TranscoderISO8859_1( const TranscoderISO8859_1 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_1( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-1"; }
     virtual TranscoderISO8859_1 *clone() const;
};

class TranscoderISO8859_2: public TranscoderISO_CP
{
public:
   TranscoderISO8859_2( const TranscoderISO8859_2 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_2( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-2"; }
     virtual TranscoderISO8859_2 *clone() const;
};

class TranscoderISO8859_3: public TranscoderISO_CP
{
public:
   TranscoderISO8859_3( const TranscoderISO8859_3 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_3( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-3"; }
     virtual TranscoderISO8859_3 *clone() const;
};


class TranscoderISO8859_4: public TranscoderISO_CP
{
public:
   TranscoderISO8859_4( const TranscoderISO8859_4 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_4( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-4"; }
     virtual TranscoderISO8859_4 *clone() const;
};

class TranscoderISO8859_5: public TranscoderISO_CP
{
public:
   TranscoderISO8859_5( const TranscoderISO8859_5 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_5( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-5"; }
     virtual TranscoderISO8859_5 *clone() const;
};

class TranscoderISO8859_6: public TranscoderISO_CP
{
public:
   TranscoderISO8859_6( const TranscoderISO8859_6 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_6( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-6"; }
     virtual TranscoderISO8859_6 *clone() const;
};

class TranscoderISO8859_7: public TranscoderISO_CP
{
public:
   TranscoderISO8859_7( const TranscoderISO8859_7 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_7( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-7"; }
     virtual TranscoderISO8859_7 *clone() const;
};

class TranscoderISO8859_8: public TranscoderISO_CP
{
public:
   TranscoderISO8859_8( const TranscoderISO8859_8 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_8( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-8"; }
     virtual TranscoderISO8859_8 *clone() const;
};

class TranscoderISO8859_9: public TranscoderISO_CP
{
public:
   TranscoderISO8859_9( const TranscoderISO8859_9 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_9( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-9"; }
     virtual TranscoderISO8859_9 *clone() const;
};

class TranscoderISO8859_10: public TranscoderISO_CP
{
public:
   TranscoderISO8859_10( const TranscoderISO8859_10 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_10( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-10"; }
     virtual TranscoderISO8859_10 *clone() const;
};

class TranscoderISO8859_11: public TranscoderISO_CP
{
public:
   TranscoderISO8859_11( const TranscoderISO8859_11 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_11( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-11"; }
     virtual TranscoderISO8859_11 *clone() const;
};

class TranscoderISO8859_13: public TranscoderISO_CP
{
public:
   TranscoderISO8859_13( const TranscoderISO8859_13 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_13( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-13"; }
     virtual TranscoderISO8859_13 *clone() const;
};

class TranscoderISO8859_14: public TranscoderISO_CP
{
public:
   TranscoderISO8859_14( const TranscoderISO8859_14 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_14( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-14"; }
     virtual TranscoderISO8859_14 *clone() const;
};

class TranscoderISO8859_15: public TranscoderISO_CP
{
public:
   TranscoderISO8859_15( const TranscoderISO8859_15 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_15( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-15"; }
     virtual TranscoderISO8859_15 *clone() const;
};

//=============================================================================
// Transcoder class
//

Transcoder::Transcoder( Stream *s, bool bOwn ):
   Stream( t_proxy ),
   m_stream( s ),
   m_parseStatus( true ),
   m_streamOwner( bOwn )
{
}

Transcoder::Transcoder( const Transcoder &other ):
   Stream( other ),
   m_parseStatus( other.m_parseStatus ),
   m_streamOwner( other.m_streamOwner )
{
   if( m_streamOwner )
      m_stream = static_cast<Stream *>(other.m_stream->clone());
   else
      m_stream = other.m_stream;
}

Transcoder::~Transcoder()
{
   if ( m_streamOwner  )
      delete m_stream;
}

void Transcoder::setUnderlying( Stream *s, bool owner )
{
   if ( m_streamOwner  )
      delete m_stream;

   m_stream = s;
   m_streamOwner = owner;
}

int64 Transcoder::seek( int64 pos, e_whence w )
{
   int64 res = m_stream->seek( pos, w );
   m_status = m_stream->status();
   return res;
}

int64 Transcoder::tell() {
   int64 res = m_stream->tell();
   m_status = m_stream->status();
   return res;
}

bool Transcoder::flush() {
   bool res = m_stream->flush();
   m_status = m_stream->status();
   return res;
}

bool Transcoder::truncate( int64 pos )
{
   bool ret = m_stream->truncate( pos );
   m_status = m_stream->status();
   return ret;
}

bool Transcoder::close()
{
   bool ret = m_stream->close();
   m_status = m_stream->status();
   return ret;
}

bool Transcoder::writeString( const String &source, uint32 begin, uint32 end )
{
   uint32 pos = begin;
   bool exitStatus = true;
   if ( end > source.length() )
      end = source.length();

   while( pos < end ) {
      // some error in writing?
      if ( ! put( source.getCharAt( pos ) ) )
         return false;

      // Status broken? -- record this and report later
      if ( exitStatus && ! m_parseStatus )
         exitStatus = false;

      ++pos;
   }

   // write is complete, but status may be broken
   m_parseStatus = exitStatus;

   return true;
}

bool Transcoder::readString( String &source, uint32 size )
{
   bool exitStatus = true;
   source.manipulator( &csh::handler_buffer );
   source.size(0);
   uint32 chr;

   while( size > 0 && get(chr) ) {

      // Status broken? -- record this and report later
      if ( exitStatus && ! m_parseStatus )
         exitStatus = false;

      source.append( chr );
      size--;
   }

   // write is complete, but status may be broken
   m_parseStatus = exitStatus;

   return exitStatus;
}


//=============================================================================
// Transparent byte oriented transcoder.
//
TranscoderByte::TranscoderByte( const TranscoderByte &other ):
   Transcoder( other ),
   m_substitute( other.m_substitute )
{}

bool TranscoderByte::get( uint32 &chr )
{
   m_parseStatus = true;

   if( popBuffer( chr ) )
      return true;

   // we may have hit eof before
   if ( m_stream->eof() )
      return false;

   byte b;
   if ( m_stream->read( &b, 1 )  == 1 )
   {
      m_parseStatus = m_stream->good();
      chr = (uint32) b;
      return true;
   }

   return false;
}

bool TranscoderByte::put( uint32 chr )
{
   m_parseStatus = true;
   byte b;
   if ( chr <= 0xFF )
   {
      b = (byte) chr;
   }
   else {
      b = m_substitute;
      m_parseStatus = false;
   }

   return ( m_stream->write( &b, 1 ) == 1 );
}

TranscoderByte *TranscoderByte::clone() const
{
   return new TranscoderByte( *this );
}

//=============================================================================
// EOL transcoder
//
TranscoderEOL::TranscoderEOL( const TranscoderEOL &other ):
   Transcoder( other )
{}

bool TranscoderEOL::get( uint32 &chr )
{
   if( popBuffer( chr ) )
      return true;

   if ( m_stream->get( chr ) )
   {
      if ( chr == (uint32) '\r' )
      {
         if ( m_stream->get( chr )  )
         {
            if ( chr != (uint32) '\n' ) {
               unget( (uint32) chr );
               chr = (uint32) '\r';
            }
         }
         else {
            chr = (uint32) '\r'; // reset chr
         }
      }
      return true;
   }

   return false;
}

bool TranscoderEOL::put( uint32 chr )
{
   if ( chr == (uint32)'\n' )
   {
      if ( m_stream->put( (uint32)'\r' ) )
         return m_stream->put( (uint32)'\n' );
      else
         return false;
   }

   return m_stream->put( chr );
}

TranscoderEOL *TranscoderEOL::clone() const
{
   return new TranscoderEOL( *this );
}


//=============================================================================
// UTF-8 Class
//

TranscoderUTF8::TranscoderUTF8( const TranscoderUTF8 &other ):
   Transcoder( other )
{}

bool TranscoderUTF8::get( uint32 &chr )
{
   m_parseStatus = true;

   if( popBuffer( chr ) )
      return true;

   byte in;
   if ( m_stream->read( &in, 1 ) != 1 )
      return false;

   // 4 bytes? -- pattern 1111 0xxx
   int count;
   if ( (in & 0xF8) == 0xF0 )
   {
      chr = (in & 0x7 ) << 18;
      count = 18;
   }
   // pattern 1110 xxxx
   else if ( (in & 0xF0) == 0xE0 )
   {
      chr = (in & 0xF) << 12;
      count = 12;
   }
   // pattern 110x xxxx
   else if ( (in & 0xE0) == 0xC0 )
   {
      chr = (in & 0x1F) << 6;
      count = 6;
   }
   else if( in < 0x80 )
   {
      chr = (uint32) in;
      return true;
   }
   // invalid pattern
   else {
      m_parseStatus = false;
      return false;
   }

   // read the other characters with pattern 0x10xx xxxx
   while( count > 0 )
   {
      count -= 6;

      int res = m_stream->read( &in, 1 );

      if ( res < 0 ) {
         // stream error?
         return false;
      }
      else if( res == 0 )
      {
         // eof before complete? -- protocol error
         m_parseStatus = false;
         return false;
      }
      else if( (in & 0xC0) != 0x80 )
      {
         // unrecognized pattern, protocol error
         m_parseStatus = false;
         return false;
      }

      chr |= (in & 0x3f) << count;
   }

   return true;
}

bool TranscoderUTF8::put( uint32 chr )
{
   m_parseStatus = true;
   byte res[4];
   uint32 resCount;

   if ( chr < 0x80 )
   {
      res[0] = (byte) chr;
      resCount = 1;
   }
   else if ( chr < 0x800 )
   {
      res[0] = 0xC0 | ((chr >> 6 ) & 0x1f);
      res[1] = 0x80 | (0x3f & chr);
      resCount = 2;
   }
   else if ( chr < 0x10000 )
   {
      res[0] = 0xE0 | ((chr >> 12) & 0x0f );
      res[1] = 0x80 | ((chr >> 6) & 0x3f );
      res[2] = 0x80 | (0x3f & chr);
      resCount = 3;
   }
   else
   {
      res[0] = 0xF0 | ((chr >> 18) & 0x7 );
      res[1] = 0x80 | ((chr >> 12) & 0x3f );
      res[2] = 0x80 | ((chr >> 6) & 0x3f );
      res[3] = 0x80 | (0x3f & chr);
      resCount = 4;
   }

   return ( m_stream->write( res, resCount ) == (int32) resCount);
}

TranscoderUTF8 *TranscoderUTF8::clone() const
{
   return new TranscoderUTF8( *this );
}

//=============================================================================
// UTF-16 Class
//

TranscoderUTF16::TranscoderUTF16( const TranscoderUTF16 &other ):
   Transcoder( other ),
   m_defEndian( other.m_defEndian ),
   m_bFirstIn( other.m_bFirstIn ),
   m_bFirstOut( other.m_bFirstOut ),
   m_bom( other.m_bom )
{}

TranscoderUTF16::TranscoderUTF16( Stream *s, bool bOwn, t_endianity endianity ):
      Transcoder( s, bOwn ),
      m_defEndian( endianity ),
      m_bFirstIn( true ),
      m_bFirstOut( true ),
      m_bom( true )
{
   // detect host endianity
   uint16 BOM = 0xFEFF;
   byte *bit = (byte *) &BOM;
   m_hostEndian = bit[0] == 0xFE ? e_be : e_le;
}

bool TranscoderUTF16::get( uint32 &chr )
{
   m_parseStatus = true;

   if( popBuffer( chr ) )
      return true;

   uint16 in;
   // first time around? read the marker.

   if( m_bFirstIn )
   {
      m_bFirstIn = false;

      // is this version of UTF working with a byte order mark?
      if ( m_bom ) {

         if ( m_stream->read( &in, 2 ) != 2 )
         {
            return false;
         }

         if( in == 0xFEFF ) {
            // local encoding
            m_streamEndian = m_hostEndian;

            if ( m_stream->read( &in, 2 ) != 2 )
               return false;
         }
         else if ( in == 0xFFFE )
         {
            // opposite encoding
            if( m_hostEndian == e_le )
               m_streamEndian = e_be;
            else
               m_streamEndian = e_le;

            if ( m_stream->read( &in, 2 ) != 2 )
               return false;
         }
         else {
            // consider stream endianity = default endianity
            if ( m_defEndian == e_detect )
               m_streamEndian = m_hostEndian;
            else
               m_streamEndian = m_defEndian;
             // and keep char.
         }
      }
      else{
         // consider stream endianity = default endianity
         if ( m_defEndian == e_detect )
            m_streamEndian = m_hostEndian;
         else
            m_streamEndian = m_defEndian;
         // and read char
         if ( m_stream->read( &in, 2 ) != 2 )
            return false;
      }
   }
   else {
    if ( m_stream->read( &in, 2 ) != 2 )
      return false;
   }

   // check endianity.
   if ( m_streamEndian != m_hostEndian )
      in = (in >> 8) | (in << 8);

   // high order char surrgoate?
   if ( in >= 0xD800 && in <= 0xDFFF )
   {
      if( in > 0xDBFF ) {
         // protocol error
         m_parseStatus = false;
         return false;
      }

      chr = (in & 0x3FF) << 10;
      int res = m_stream->read( &in, 2 );
      if ( res == -1 )
            return false;
      else if ( res != 2 )
      {
         // eof before
         m_parseStatus = false;
         return false;
      }

      // endianity
      if ( m_streamEndian != m_hostEndian )
         in = (in >> 8) | (in << 8);
      // protocol check
      if( in < 0xDC00 || in > 0xDFFF )
      {
         m_parseStatus = false;
         return false;
      }

      // fine, we can complete.
      chr |= (in & 0x3FF);
   }
   else
      chr = (uint32) in;

   return true;
}

bool TranscoderUTF16::put( uint32 chr )
{
   uint16 out;

   if ( m_bFirstOut )
   {
      m_bFirstOut = false;
      if ( m_bom )
      {
         // write byte order marker
         out = 0xFEFF;
         // consider stream endianity = default endianity
         if ( m_defEndian == e_detect ) {
            m_streamEndian = m_hostEndian;
         }
         else {
            m_streamEndian = m_defEndian;
            if( m_streamEndian != m_hostEndian )
               out = (out >> 8 ) | ( out << 8 );
         }

         if ( m_stream->write( &out, 2 ) != 2 )
            return false;
      }
      else {
         // just set endianity
         if ( m_defEndian == e_detect ) {
            m_streamEndian = m_hostEndian;
         }
         else {
            m_streamEndian = m_defEndian;
         }
      }
   }

   if ( chr < 0x10000 )
   {
      out = (uint16) chr;
      if( m_streamEndian != m_hostEndian )
         out = (out >> 8 ) | ( out << 8 );
      if ( m_stream->write( &out, 2 ) != 2 )
         return false;
   }
   else {
      out = 0xD800 | ((chr >> 10) & 0x3FF);
      if( m_streamEndian != m_hostEndian )
         out = (out >> 8 ) | ( out << 8 );
      if ( m_stream->write( &out, 2 ) != 2 )
         return false;

      out = 0xDC00 | ((chr >> 10) & 0x3FF);
      if( m_streamEndian != m_hostEndian )
         out = (out >> 8 ) | ( out << 8 );
      if ( m_stream->write( &out, 2 ) != 2 )
         return false;
   }

   return true;
}

TranscoderUTF16 *TranscoderUTF16::clone() const
{
   return new TranscoderUTF16( *this );
}



//=========================================
// TABLES
//
TranscoderISO_CP::TranscoderISO_CP( const TranscoderISO_CP &other ):
   Transcoder( other ),
   m_directTable( other.m_directTable ),
   m_dirTabSize( other.m_dirTabSize ),
   m_reverseTable( other.m_reverseTable ),
   m_revTabSize( other.m_revTabSize ),
   m_nMinUntl( other.m_nMinUntl ),
   m_nMaxUntl( other.m_nMaxUntl )
{}

bool TranscoderISO_CP::get( uint32 &chr )
{
   m_parseStatus = true;

   if( popBuffer( chr ) )
      return true;

   // converting the character into an unicode.
   byte in;
   if ( m_stream->read( &in, 1 ) != 1 )
      return false;

   if ( in < 0xA0 )
   {
      chr = (uint32) in;
      return true;
   }

   if ( in >= 0xA0 + m_dirTabSize )
   {
      chr = (uint32) '?';
      m_parseStatus = false;
      return true;
   }

   chr = (uint32) m_directTable[ in - 0xA0 ];
   if ( chr == 0 ) {
      chr = (uint32) '?';
      m_parseStatus = false;
   }

   return true;
}

bool TranscoderISO_CP::put( uint32 chr )
{
   m_parseStatus = true;
   byte out;
   if ( chr >= m_nMinUntl && chr <= m_nMaxUntl )
   {
      out = (byte) chr;
   }
   else {
      // scan the table.
      uint32 lower = 0;
      uint32 higher = m_revTabSize-1;
      uint32 point = higher / 2;
      uint32 val;

      while ( true )
      {
         // get the table row
         val = (uint32) m_reverseTable[ point ].unicode;

         if ( val == chr )
         {
            out = (byte) m_reverseTable[ point ].local;
            break;
         }
         else
         {
            if ( lower == higher )  // not found
            {
               out = (byte) '?';
               m_parseStatus = false;
               break;
            }
            // last try. In pair sized dictionaries, it can be also in the other node
            else if ( lower == higher -1 )
            {
               point = lower = higher;
               continue;
            }
         }

         if ( chr > val )
         {
            lower = point;
         }
         else
         {
            higher = point;
         }
         point = ( lower + higher ) / 2;
      }
   }

   // write the result
   return (m_stream->write( &out, 1 ) == 1);
}

TranscoderCP1252::TranscoderCP1252( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_cp1252;
   m_dirTabSize = sizeof( s_table_cp1252 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_cp1252;
   m_revTabSize = sizeof( s_rtable_cp1252 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderCP1252 *TranscoderCP1252::clone() const
{
   return new TranscoderCP1252( *this );
}



TranscoderISO8859_1::TranscoderISO8859_1( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_1;
   m_dirTabSize = sizeof( s_table_iso8859_1 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_1;
   m_revTabSize = sizeof( s_rtable_iso8859_1 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_1 *TranscoderISO8859_1::clone() const
{
   return new TranscoderISO8859_1( *this );
}

TranscoderISO8859_2::TranscoderISO8859_2( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_2;
   m_dirTabSize = sizeof( s_table_iso8859_2 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_2;
   m_revTabSize = sizeof( s_rtable_iso8859_2 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_2 *TranscoderISO8859_2::clone() const
{
   return new TranscoderISO8859_2( *this );
}

TranscoderISO8859_3::TranscoderISO8859_3( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_3;
   m_dirTabSize = sizeof( s_table_iso8859_3 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_3;
   m_revTabSize = sizeof( s_rtable_iso8859_3 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_3 *TranscoderISO8859_3::clone() const
{
   return new TranscoderISO8859_3( *this );
}


TranscoderISO8859_4::TranscoderISO8859_4( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_4;
   m_dirTabSize = sizeof( s_table_iso8859_4 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_4;
   m_revTabSize = sizeof( s_rtable_iso8859_4 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_4 *TranscoderISO8859_4::clone() const
{
   return new TranscoderISO8859_4( *this );
}


TranscoderISO8859_5::TranscoderISO8859_5( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_5;
   m_dirTabSize = sizeof( s_table_iso8859_5 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_5;
   m_revTabSize = sizeof( s_rtable_iso8859_5 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_5 *TranscoderISO8859_5::clone() const
{
   return new TranscoderISO8859_5( *this );
}


TranscoderISO8859_6::TranscoderISO8859_6( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_6;
   m_dirTabSize = sizeof( s_table_iso8859_6 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_6;
   m_revTabSize = sizeof( s_rtable_iso8859_6 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_6 *TranscoderISO8859_6::clone() const
{
   return new TranscoderISO8859_6( *this );
}

TranscoderISO8859_7::TranscoderISO8859_7( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_7;
   m_dirTabSize = sizeof( s_table_iso8859_7 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_7;
   m_revTabSize = sizeof( s_rtable_iso8859_7 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_7 *TranscoderISO8859_7::clone() const
{
   return new TranscoderISO8859_7( *this );
}

TranscoderISO8859_8::TranscoderISO8859_8( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_8;
   m_dirTabSize = sizeof( s_table_iso8859_8 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_8;
   m_revTabSize = sizeof( s_rtable_iso8859_8 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_8 *TranscoderISO8859_8::clone() const
{
   return new TranscoderISO8859_8( *this );
}

TranscoderISO8859_9::TranscoderISO8859_9( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_9;
   m_dirTabSize = sizeof( s_table_iso8859_9 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_9;
   m_revTabSize = sizeof( s_rtable_iso8859_9 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_9 *TranscoderISO8859_9::clone() const
{
   return new TranscoderISO8859_9( *this );
}


TranscoderISO8859_10::TranscoderISO8859_10( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_10;
   m_dirTabSize = sizeof( s_table_iso8859_10 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_10;
   m_revTabSize = sizeof( s_rtable_iso8859_10 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_10 *TranscoderISO8859_10::clone() const
{
   return new TranscoderISO8859_10( *this );
}

TranscoderISO8859_11::TranscoderISO8859_11( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_11;
   m_dirTabSize = sizeof( s_table_iso8859_11 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_11;
   m_revTabSize = sizeof( s_rtable_iso8859_11 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_11 *TranscoderISO8859_11::clone() const
{
   return new TranscoderISO8859_11( *this );
}

TranscoderISO8859_13::TranscoderISO8859_13( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_13;
   m_dirTabSize = sizeof( s_table_iso8859_13 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_13;
   m_revTabSize = sizeof( s_rtable_iso8859_13 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_13 *TranscoderISO8859_13::clone() const
{
   return new TranscoderISO8859_13( *this );
}

TranscoderISO8859_14::TranscoderISO8859_14( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_14;
   m_dirTabSize = sizeof( s_table_iso8859_14 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_14;
   m_revTabSize = sizeof( s_rtable_iso8859_14 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_14 *TranscoderISO8859_14::clone() const
{
   return new TranscoderISO8859_14( *this );
}

TranscoderISO8859_15::TranscoderISO8859_15( Stream *s, bool bOwn ):
   TranscoderISO_CP( s, bOwn )
{
   m_directTable = s_table_iso8859_15;
   m_dirTabSize = sizeof( s_table_iso8859_15 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_iso8859_15;
   m_revTabSize = sizeof( s_rtable_iso8859_15 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderISO8859_15 *TranscoderISO8859_15::clone() const
{
   return new TranscoderISO8859_15( *this );
}


//=========================================
// OEM tables (covering 0x0 to 0x20  and 0x80 to 0xff ranges)
//

TranscoderOEM::TranscoderOEM( const TranscoderOEM &other ):
   TranscoderISO_CP( other )
{}

bool TranscoderOEM::get( uint32 &chr )
{
   m_parseStatus = true;

   if( popBuffer( chr ) )
      return true;

   // converting the character into an unicode.
   byte in;
   if ( m_stream->read( &in, 1 ) != 1 )
      return false;

   if ( in >= m_nMinUntl && in <= m_nMaxUntl )
   {
      chr = (uint32) in;
      return true;
   }

   if ( in >= 0xFF )
   {
      chr = (uint32) '?';
      m_parseStatus = false;
      return true;
   }

   chr = chr < m_nMinUntl ? 
      (uint32) m_directTable[ in ] :
      (uint32) m_directTable[ in + m_nMinUntl - m_nMaxUntl-1 ];

   return true;
}

TranscoderIBM437::TranscoderIBM437( Stream *s, bool bOwn ):
   TranscoderOEM( s, bOwn )
{
   m_directTable = s_table_IBM437;
   m_dirTabSize = sizeof( s_table_IBM437 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_IBM437;
   m_revTabSize = sizeof( s_rtable_IBM437 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderIBM437 *TranscoderIBM437::clone() const
{
   return new TranscoderIBM437( *this );
}


TranscoderIBM850::TranscoderIBM850( Stream *s, bool bOwn ):
   TranscoderOEM( s, bOwn )
{
   m_directTable = s_table_IBM850;
   m_dirTabSize = sizeof( s_table_IBM850 ) / sizeof( uint16 );
   m_reverseTable = s_rtable_IBM850;
   m_revTabSize = sizeof( s_rtable_IBM850 ) / sizeof( CP_ISO_UINT_TABLE );
}

TranscoderIBM850 *TranscoderIBM850::clone() const
{
   return new TranscoderIBM850( *this );
}

//==================================================================
// Utilities

Transcoder *TranscoderFactory( const String &encoding, Stream *stream, bool own )
{

   if ( encoding == "C" )
      return new TranscoderByte( stream, own );

   if ( encoding == "utf-8" )
      return new TranscoderUTF8( stream, own );

   if ( encoding == "utf-16" )
      return new TranscoderUTF16( stream, own );

   if ( encoding == "utf-16LE" )
      return new TranscoderUTF16LE( stream, own );

   if ( encoding == "utf-16BE" )
      return new TranscoderUTF16BE( stream, own );

   if ( encoding == "cp1252" )
      return new TranscoderCP1252( stream, own );

   if ( encoding == "IBM850" )
      return new TranscoderIBM850( stream, own );

   if ( encoding == "IBM437" )
      return new TranscoderIBM437( stream, own );

   if ( encoding == "iso8859-1" )
      return new TranscoderISO8859_1( stream, own );

   if ( encoding == "iso8859-2" )
      return new TranscoderISO8859_2( stream, own );

   if ( encoding == "iso8859-3" )
      return new TranscoderISO8859_3( stream, own );

   if ( encoding == "iso8859-4" )
      return new TranscoderISO8859_4( stream, own );

   if ( encoding == "iso8859-5" )
      return new TranscoderISO8859_5( stream, own );

   if ( encoding == "iso8859-6" )
      return new TranscoderISO8859_6( stream, own );

   if ( encoding == "iso8859-7" )
      return new TranscoderISO8859_7( stream, own );

   if ( encoding == "iso8859-8" )
      return new TranscoderISO8859_8( stream, own );

   if ( encoding == "iso8859-9" )
      return new TranscoderISO8859_9( stream, own );

   if ( encoding == "iso8859-10" )
      return new TranscoderISO8859_10( stream, own );

   if ( encoding == "iso8859-11" )
      return new TranscoderISO8859_11( stream, own );

   // in case you are wondering, iso8859-12 has never been defined.

   if ( encoding == "iso8859-13" )
      return new TranscoderISO8859_13( stream, own );

   if ( encoding == "iso8859-14" )
      return new TranscoderISO8859_14( stream, own );

   if ( encoding == "iso8859-15" )
      return new TranscoderISO8859_15( stream, own );

   if ( encoding == "gbk" )
      return new TranscoderGBK( stream, own );
   return 0;
}

bool TranscodeString( const String &source, const String &encoding, String &target )
{
   Transcoder *tr = TranscoderFactory( encoding );
   if ( tr == 0 )
      return false;

   StringStream output;
   tr->setUnderlying( &output );
   tr->writeString( source );
   output.closeToString( target );
   delete tr;
   return true;
}

bool TranscodeFromString( const String &source, const String &encoding, String &target )
{
   Transcoder *tr = TranscoderFactory( encoding );
   if ( tr == 0 )
      return false;

   StringStream input;
   input.writeString( source );
   input.seekBegin( 0 );
   tr->setUnderlying( &input );
   tr->readString( target, 4096 );
   while( tr->good() && ! tr->eof() )
   {
      String temp;
      tr->readString( temp, 4096 );
      target += temp;
   }
   delete tr;
   return true;
}

#ifdef WIN32
bool GetSystemEncodingWin(String &encoding)
{
   UINT ConsoleInputCP = GetConsoleCP();
   switch (ConsoleInputCP)
   {
   case 37: encoding = "IBM037";	return false;   // IBM EBCDIC US-Canada
   case 437: encoding = "IBM437";	return true;   // OEM United States
   case 500: encoding = "IBM500";	return false; 	// IBM EBCDIC International
   case 708: encoding = "ASMO-708";	return false; 	// Arabic (ASMO 708)
   case 709: encoding = "";	return false; 	// Arabic (ASMO-449+, BCON V4)
   case 710: encoding = "";	return false; 	// Arabic - Transparent Arabic
   case 720: encoding = "DOS-720";	return false; 	// Arabic (Transparent ASMO); Arabic (DOS)
   case 737: encoding = "IBM737";	return false; 	// OEM Greek (formerly 437G); Greek (DOS)
   case 775: encoding = "IBM775";	return false; 	// OEM Baltic; Baltic (DOS)
   case 850: encoding = "IBM850";	return true; 	// OEM Multilingual Latin 1; Western European (DOS)
   case 852: encoding = "IBM852";	return false; 	//	OEM Latin 2; Central European (DOS)
   case 855: encoding = "IBM855";	return false; 	//	OEM Cyrillic (primarily Russian)
   case 857: encoding = "IBM857";	return false; 	//	OEM Turkish; Turkish (DOS)
   case 858: encoding = "IBM00858";	return false; 	//	OEM Multilingual Latin 1 + Euro symbol
   case 860: encoding = "IBM860";	return false; 	//	OEM Portuguese; Portuguese (DOS)
   case 861: encoding = "IBM861";	return false; 	//	OEM Icelandic; Icelandic (DOS)
   case 862: encoding = "DOS-862";	return false; 	//	OEM Hebrew; Hebrew (DOS)
   case 863: encoding = "IBM863";	return false; 	//	OEM French Canadian; French Canadian (DOS)
   case 864: encoding = "IBM864";	return false; 	//	OEM Arabic; Arabic (864)
   case 865: encoding = "IBM865";	return false; 	//	OEM Nordic; Nordic (DOS)
   case 866: encoding = "cp866";	return false; 	//	OEM Russian; Cyrillic (DOS)
   case 869: encoding = "IBM869";	return false; 	//	OEM Modern Greek; Greek, Modern (DOS)
   case 870: encoding = "IBM870";	return false; 	//	IBM EBCDIC Multilingual/ROECE (Latin 2); IBM EBCDIC Multilingual Latin 2
   case 874: encoding = "windows-874";	return false; 	//	ANSI/OEM Thai (same as 28605, ISO 8859-15); Thai (Windows)
   case 875: encoding = "cp875";	return false; 	//	IBM EBCDIC Greek Modern
   case 932: encoding = "shift_jis";	return false; 	//	ANSI/OEM Japanese; Japanese (Shift-JIS)
   //case 936: encoding = "gb2312";	return false; 	//	ANSI/OEM Simplified Chinese (PRC, Singapore); Chinese Simplified (GB2312)
   //TODO: must select a properly name for GBK; such as MS936, cp936 is too general, 
   case 936: encoding = "gbk";	return true; 	//	ANSI/OEM Simplified Chinese (PRC, Singapore); Chinese Simplified (GB2312)
   case 949: encoding = "ks_c_5601-1987";	return false; 	//	ANSI/OEM Korean (Unified Hangul Code)
   case 950: encoding = "big5	ANSI/OEM";	return false; 	// Traditional Chinese (Taiwan; Hong Kong SAR, PRC); Chinese Traditional (Big5)
   case 1026: encoding = "IBM1026";	return false; 	//	IBM EBCDIC Turkish (Latin 5)
   case 1047: encoding = "IBM01047";	return false; 	//	IBM EBCDIC Latin 1/Open System
   case 1140: encoding = "IBM01140";	return false; 	//	IBM EBCDIC US-Canada (037 + Euro symbol); IBM EBCDIC (US-Canada-Euro)
   case 1141: encoding = "IBM01141";	return false; 	//	IBM EBCDIC Germany (20273 + Euro symbol); IBM EBCDIC (Germany-Euro)
   case 1142: encoding = "IBM01142";	return false; 	//	IBM EBCDIC Denmark-Norway (20277 + Euro symbol); IBM EBCDIC (Denmark-Norway-Euro)
   case 1143: encoding = "IBM01143";	return false; 	//	IBM EBCDIC Finland-Sweden (20278 + Euro symbol); IBM EBCDIC (Finland-Sweden-Euro)
   case 1144: encoding = "IBM01144";	return false; 	//	IBM EBCDIC Italy (20280 + Euro symbol); IBM EBCDIC (Italy-Euro)
   case 1145: encoding = "IBM01145";	return false; 	//	IBM EBCDIC Latin America-Spain (20284 + Euro symbol); IBM EBCDIC (Spain-Euro)
   case 1146: encoding = "IBM01146";	return false; 	//	IBM EBCDIC United Kingdom (20285 + Euro symbol); IBM EBCDIC (UK-Euro)
   case 1147: encoding = "IBM01147";	return false; 	//	IBM EBCDIC France (20297 + Euro symbol); IBM EBCDIC (France-Euro)
   case 1148: encoding = "IBM01148";	return false; 	//	IBM EBCDIC International (500 + Euro symbol); IBM EBCDIC (International-Euro)
   case 1149: encoding = "IBM01149";	return false; 	//	IBM EBCDIC Icelandic (20871 + Euro symbol); IBM EBCDIC (Icelandic-Euro)
   case 1200: encoding = "utf-16LE";	return true; 	//	Unicode UTF-16, little endian byte order (BMP of ISO 10646); available only to managed applications
   case 1201: encoding = "utf-16BE";	return false; 	//	Unicode UTF-16, big endian byte order; available only to managed applications
   case 1250: encoding = "windows-1250";	return false; 	//	ANSI Central European; Central European (Windows)
   case 1251: encoding = "windows-1251";	return false; 	//	ANSI Cyrillic; Cyrillic (Windows)
   case 1252: encoding = "cp1252";	return true; 	//	ANSI Latin 1; Western European (Windows)
   case 1253: encoding = "windows-1253";	return false; 	//	ANSI Greek; Greek (Windows)
   case 1254: encoding = "windows-1254";	return false; 	//	ANSI Turkish; Turkish (Windows)
   case 1255: encoding = "windows-1255";	return false; 	//	ANSI Hebrew; Hebrew (Windows)
   case 1256: encoding = "windows-1256";	return false; 	//	ANSI Arabic; Arabic (Windows)
   case 1257: encoding = "windows-1257";	return false; 	//	ANSI Baltic; Baltic (Windows)
   case 1258: encoding = "windows-1258";	return false; 	//	ANSI/OEM Vietnamese; Vietnamese (Windows)
   case 1361: encoding = "";	return false; 	//Johab	Korean (Johab)
   case 10000: encoding = "macintosh";	return false; 	//	MAC Roman; Western European (Mac)
   case 10001: encoding = "x-mac-japanese	Japanese";	return false; 	// (Mac)
   case 10002: encoding = "x-mac-chinesetrad";	return false; 	//	MAC Traditional Chinese (Big5); Chinese Traditional (Mac)
   case 10003: encoding = "x-mac-korean";	return false; 	//	Korean (Mac)
   case 10004: encoding = "x-mac-arabic";	return false; 	//	Arabic (Mac)
   case 10005: encoding = "x-mac-hebrew";	return false; 	//	Hebrew (Mac)
   case 10006: encoding = "x-mac-greek";	return false; 	//	Greek (Mac)
   case 10007: encoding = "x-mac-cyrillic";	return false; 	//	Cyrillic (Mac)
   case 10008: encoding = "x-mac-chinesesimp";	return false; 	//	MAC Simplified Chinese (GB 2312); Chinese Simplified (Mac)
   case 10010: encoding = "x-mac-romanian";	return false; 	//	Romanian (Mac)
   case 10017: encoding = "x-mac-ukrainian";	return false; 	//	Ukrainian (Mac)
   case 10021: encoding = "x-mac-thai";	return false; 	//	Thai (Mac)
   case 10029: encoding = "x-mac-ce";	return false; 	//	MAC Latin 2; Central European (Mac)
   case 10079: encoding = "x-mac-icelandic";	return false; 	//	Icelandic (Mac)
   case 10081: encoding = "x-mac-turkish";	return false; 	//	Turkish (Mac)
   case 10082: encoding = "x-mac-croatian";	return false; 	//	Croatian (Mac)
   case 12000: encoding = "utf-32";	return false; 	//	Unicode UTF-32, little endian byte order; available only to managed applications
   case 12001: encoding = "utf-32BE";	return false; 	//	Unicode UTF-32, big endian byte order; available only to managed applications
   case 20000: encoding = "x-Chinese_CNS";	return false; 	//	CNS Taiwan; Chinese Traditional (CNS)
   case 20001: encoding = "x-cp20001";	return false; 	//	TCA Taiwan
   case 20002: encoding = "x_Chinese-Eten";	return false; 	//	Eten Taiwan; Chinese Traditional (Eten)
   case 20003: encoding = "x-cp20003";	return false; 	//	IBM5550 Taiwan
   case 20004: encoding = "x-cp20004";	return false; 	//	TeleText Taiwan
   case 20005: encoding = "x-cp20005";	return false; 	//	Wang Taiwan
   case 20105: encoding = "x-IA5	IA5";	return false; 	// (IRV International Alphabet No. 5, 7-bit); Western European (IA5)
   case 20106: encoding = "x-IA5-German";	return false; 	//	IA5 German (7-bit)
   case 20107: encoding = "x-IA5-Swedish";	return false; 	//	IA5 Swedish (7-bit)
   case 20108: encoding = "x-IA5-Norwegian";	return false; 	//	IA5 Norwegian (7-bit)
   case 20127: encoding = "us-ascii";	return false; 	//	US-ASCII (7-bit)
   case 20261: encoding = "x-cp20261";	return false; 	//	T.61
   case 20269: encoding = "x-cp20269";	return false; 	//	ISO 6937 Non-Spacing Accent
   case 20273: encoding = "IBM273";	return false; 	//	IBM EBCDIC Germany
   case 20277: encoding = "IBM277";	return false; 	//	IBM EBCDIC Denmark-Norway
   case 20278: encoding = "IBM278";	return false; 	//	IBM EBCDIC Finland-Sweden
   case 20280: encoding = "IBM280";	return false; 	//	IBM EBCDIC Italy
   case 20284: encoding = "IBM284";	return false; 	//	IBM EBCDIC Latin America-Spain
   case 20285: encoding = "IBM285";	return false; 	//	IBM EBCDIC United Kingdom
   case 20290: encoding = "IBM290";	return false; 	//	IBM EBCDIC Japanese Katakana Extended
   case 20297: encoding = "IBM297";	return false; 	//	IBM EBCDIC France
   case 20420: encoding = "IBM420";	return false; 	//	IBM EBCDIC Arabic
   case 20423: encoding = "IBM423";	return false; 	//	IBM EBCDIC Greek
   case 20424: encoding = "IBM424";	return false; 	//	IBM EBCDIC Hebrew
   case 20833: encoding = "x-EBCDIC-KoreanExtended";	return false; 	//	IBM EBCDIC Korean Extended
   case 20838: encoding = "IBM-Thai";	return false; 	//	IBM EBCDIC Thai
   case 20866: encoding = "koi8-r";	return false; 	//	Russian (KOI8-R); Cyrillic (KOI8-R)
   case 20871: encoding = "IBM871";	return false; 	//	IBM EBCDIC Icelandic
   case 20880: encoding = "IBM880";	return false; 	//	IBM EBCDIC Cyrillic Russian
   case 20905: encoding = "IBM905";	return false; 	//	IBM EBCDIC Turkish
   case 20924: encoding = "IBM00924";	return false; 	//	IBM EBCDIC Latin 1/Open System (1047 + Euro symbol)
   case 20932: encoding = "EUC-JP";	return false; 	//	Japanese (JIS 0208-1990 and 0121-1990)
   case 20936: encoding = "x-cp20936";	return false; 	//	Simplified Chinese (GB2312); Chinese Simplified (GB2312-80)
   case 20949: encoding = "x-cp20949";	return false; 	//	Korean Wansung
   case 21025: encoding = "cp1025";	return false; 	//	IBM EBCDIC Cyrillic Serbian-Bulgarian
   case 21027: encoding = "";	return false; 	//	(deprecated)
   case 21866: encoding = "koi8-u";	return false; 	//	Ukrainian (KOI8-U); Cyrillic (KOI8-U)
   case 28591: encoding = "iso8859-1";	return true; 	//	ISO 8859-1 Latin 1; Western European (ISO)
   case 28592: encoding = "iso8859-2";	return true; 	//	ISO 8859-2 Central European; Central European (ISO)
   case 28593: encoding = "iso8859-3";	return true; 	//	ISO 8859-3 Latin 3
   case 28594: encoding = "iso8859-4";	return true; 	//	ISO 8859-4 Baltic
   case 28595: encoding = "iso8859-5";	return true; 	//	ISO 8859-5 Cyrillic
   case 28596: encoding = "iso8859-6";	return true; 	//	ISO 8859-6 Arabic
   case 28597: encoding = "iso8859-7";	return true; 	//	ISO 8859-7 Greek
   case 28598: encoding = "iso8859-8";	return true; 	//	ISO 8859-8 Hebrew; Hebrew (ISO-Visual)
   case 28599: encoding = "iso8859-9";	return true; 	//	ISO 8859-9 Turkish
   case 28603: encoding = "iso8859-13";	return true; 	//	ISO 8859-13 Estonian
   case 28605: encoding = "iso8859-15";	return true; 	//	ISO 8859-15 Latin 9
   case 29001: encoding = "x-Europa";	return false; 	//	Europa 3
   case 38598: encoding = "iso-8859-8-i";	return false; 	//	ISO 8859-8 Hebrew; Hebrew (ISO-Logical)
   case 50220: encoding = "iso-2022-jp";	return false; 	//	ISO 2022 Japanese with no halfwidth Katakana; Japanese (JIS)
   case 50221: encoding = "csISO2022JP";	return false; 	//	ISO 2022 Japanese with halfwidth Katakana; Japanese (JIS-Allow 1 byte Kana)
   case 50222: encoding = "iso-2022-jp";	return false; 	//	ISO 2022 Japanese JIS X 0201-1989; Japanese (JIS-Allow 1 byte Kana - SO/SI)
   case 50225: encoding = "iso-2022-kr";	return false; 	//	ISO 2022 Korean
   case 50227: encoding = "x-cp50227";	return false; 	//	ISO 2022 Simplified Chinese; Chinese Simplified (ISO 2022)
   case 50229: encoding = "";	return false; 	//	ISO 2022 Traditional Chinese
   case 50930: encoding = "";	return false; 	//	EBCDIC Japanese (Katakana) Extended
   case 50931: encoding = "";	return false; 	//	EBCDIC US-Canada and Japanese
   case 50933: encoding = "";	return false; 	//	EBCDIC Korean Extended and Korean
   case 50935: encoding = "";	return false; 	//	EBCDIC Simplified Chinese Extended and Simplified Chinese
   case 50936: encoding = "";	return false; 	//	EBCDIC Simplified Chinese
   case 50937: encoding = "";	return false; 	//	EBCDIC US-Canada and Traditional Chinese
   case 50939: encoding = "";	return false; 	//	EBCDIC Japanese (Latin) Extended and Japanese
   case 51932: encoding = "euc-jp";	return false; 	//	EUC Japanese
   case 51936: encoding = "EUC-CN";	return false; 	//	EUC Simplified Chinese; Chinese Simplified (EUC)
   case 51949: encoding = "euc-kr";	return false; 	//	EUC Korean
   case 51950: encoding = "";	return false; 	//	EUC Traditional Chinese
   case 52936: encoding = "hz-gb-2312";	return false; 	//	HZ-GB2312 Simplified Chinese; Chinese Simplified (HZ)
   case 54936: encoding = "GB18030";	return false; 	//	Windows XP and later: encoding = "GB18030 Simplified Chinese (4 byte); Chinese Simplified (GB18030)
   case 57002: encoding = "x-iscii-de";	return false; 	//	ISCII Devanagari
   case 57003: encoding = "x-iscii-be";	return false; 	//	ISCII Bengali
   case 57004: encoding = "x-iscii-ta";	return false; 	//	ISCII Tamil
   case 57005: encoding = "x-iscii-te";	return false; 	//	ISCII Telugu
   case 57006: encoding = "x-iscii-as";	return false; 	//	ISCII Assamese
   case 57007: encoding = "x-iscii-or";	return false; 	//	ISCII Oriya
   case 57008: encoding = "x-iscii-ka";	return false; 	//	ISCII Kannada
   case 57009: encoding = "x-iscii-ma";	return false; 	//	ISCII Malayalam
   case 57010: encoding = "x-iscii-gu";	return false; 	//	ISCII Gujarati
   case 57011: encoding = "x-iscii-pa";	return false; 	//	ISCII Punjabi
   case 65000: encoding = "utf-7";	return false; 	//	Unicode (UTF-7)
   case 65001: encoding = "utf-8";	return true; 	//	Unicode (UTF-8)
   default: encoding = ""; return false;
   }
}
#endif


bool GetSystemEncoding( String &encoding )
{
#ifdef WIN32
   return GetSystemEncodingWin(encoding);
#else
   const char *lc_ctype = setlocale( LC_CTYPE, 0 );
   if ( lc_ctype == 0 )
      return false;

   encoding = lc_ctype;
   if ( encoding == "C" || encoding == "POSIX" )
   {
      // this is a request to use only non-international strings.
      // but first see if the program has not set a locale.
      String language;
      if ( ! Sys::_getEnv( "LANG", language ) )
      {
         Sys::_getEnv( "LC_ALL", language );
      }

      if( language != "" )
      {
         encoding = language;
         if ( encoding == "C" || encoding == "POSIX" )
            return false;
         // else continue processing.
      }
      else
         return false;
   }

   //do we have an encoding specification?
   uint32 pos = encoding.find(".");
   if( pos != csh::npos && pos < encoding.length() - 1 )
   {
      // we have a dot.
      encoding = encoding.subString( pos + 1 );

      // eventually remove "@"
      pos = encoding.find( "@" );
      if ( pos != csh::npos )
      {
         encoding = encoding.subString( 0, pos );
      }

      pos = encoding.find( "8859-" );
      if ( pos != csh::npos && pos < encoding.length() - 6)
      {
         // a subkind of iso encoding.
         encoding = encoding.subString( pos + 6 );
         if ( encoding == "1" || encoding == "2" || encoding == "3" || encoding == "4" ||
            encoding == "5" || encoding == "6" || encoding == "7" || encoding == "7" ||
            encoding == "8" || encoding == "9" || encoding == "10" || encoding == "11" ||
            encoding == "13" || encoding == "14" || encoding == "15"
            )
         {
            encoding = "iso8859-" + encoding;
            return true;
         }
      }
      else {
         // utf variant?
         pos = encoding.find( "UTF" );
         if ( pos == csh::npos )
            pos = encoding.find( "utf" );
         
         uint32 utf_prefix_len=3; 
         if(pos != csh::npos && encoding.getCharAt(utf_prefix_len) == '-' )
           utf_prefix_len++;;
         
         if ( pos != csh::npos && pos < encoding.length() - utf_prefix_len )
         {
            encoding = encoding.subString( pos + utf_prefix_len );
            if ( encoding == "8" || encoding == "16" || encoding == "16LE" || encoding == "16BE" )
            {
               encoding = "utf-" + encoding;
               return true;
            }
         }
         else
         {
            // known windows cp?
            if( encoding == "1252" )
            {
               encoding = "cp" + encoding;
               return true;
            }
         }
      }

      // if we are not returned up to date, surrender.
      return false;
   }

   // no dot. Guess one from the language.
   if ( encoding.length() >= 2 )
   {
      encoding = encoding.subString( 0, 2 );
      if ( encoding == "it" || encoding == "fr" || encoding == "de" ||
          encoding == "es" || encoding == "pt" )
      {
         encoding = "iso8859-15";
         return true;
      }

      if ( encoding == "en" )
      {
         encoding = "iso8859-1";
         return true;
      }

      if ( encoding == "gr" )
      {
         encoding = "iso8859-7";
         return true;
      }
   }
   // no way
   return false;
#endif
}

}

/* end of transcoding.cpp */

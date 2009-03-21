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

/** UTF-8 encoding transcoder. */
class TranscoderUTF8: public Transcoder
{
public:
   TranscoderUTF8( Stream *s, bool bOwn=false ):
     Transcoder( s, bOwn )
   {}

   TranscoderUTF8( const TranscoderUTF8 &other );

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "utf-8"; }
   virtual FalconData *clone() const;
};

/** UTF-16 encoding transcoder. */
class TranscoderUTF16: public Transcoder
{
public:
   /** Endianity specification, see the constructor. */
   typedef enum {
      e_detect,
      e_le,
      e_be
   } t_endianity;

private:
   t_endianity m_defEndian;
   t_endianity m_streamEndian;
   t_endianity m_hostEndian;

   bool m_bFirstIn;
   bool m_bFirstOut;

protected:
   bool m_bom;

public:

   /** Constructor
   UTF-16 requires specification of endianity with a prefix character.

   On output transcoding, this character is written right before the
   first write() is performed, while on input it is read right before
   the first get() is performed.

   This constructor allows to select an endianity used in I/O. The
   default is e_none; in this case, the transcoder will detect and
   use the system endianity on output.

   In input, the marker read at first get() determines the endianity.
   If the marker can't be found (i.e. if the stream was not at beginning),
   the supplied parameter will be used.

   If the endianity is e_detect, the host system endianity is used.

   The method endianity() can be used to determine the endianity
   of the stream in input that was decided either by reading the marker
   or by the constructor.

   \param s the underlying stream
   \param bOwn own the underlying stream
   \param endianity the endianity for operations.
   */
   TranscoderUTF16( Stream *s, bool bOwn=false, t_endianity endianity = e_detect );
   TranscoderUTF16( const TranscoderUTF16 &other );

   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
   virtual const String encoding() const { return "utf-16"; }
   t_endianity endianity() const { return m_streamEndian; }
   virtual FalconData *clone() const;
};

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




typedef signed char sbyte;
class TranscoderGBK: public Transcoder
{

   /*
   * 2nd level index, provided by subclass
   * every string has 0x10*(end-start+1) characters.
   */

   const static uint32 REPLACE_CHAR = 0xFFFD;
   Table *decoderTable1;
   Table *decoderTable2;
   Table *encoderTable1;
   Table *encoderTable2;
   const static int  start = 0x40;
   const static int  end =  0xFE;
public:
   TranscoderGBK( const TranscoderGBK &other ):
      Transcoder( other ),
         decoderTable1( other.decoderTable1 ),
         decoderTable2( other.decoderTable2 ),
         encoderTable1( other.encoderTable1 ),
         encoderTable2( other.encoderTable2 )
      {}

   TranscoderGBK( Stream *s, bool bOwn=false ):
      Transcoder(s, bOwn)
      {
         decoderTable1 = &gbkDecoderTable1;
         decoderTable2 = &gbkDecoderTable2;
         encoderTable1 = &gbkEncoderTable1;
         encoderTable2 = &gbkEncoderTable2;
      }

      uint16 getUint16(Table *t, uint32 pos)
      {
         return ((uint16*)t->table)[pos];
      }

      uint32 getUTF16Char(uint16 *t, uint32 pos)
      {
         return t[pos];
      }
      Table * getTable(Table *t, uint32 pos)
      {
         return (Table*)t->table;
      }

      /*
      protected char decodeSingle(int b) {
      if (b >= 0)
      return (char) b;
      return REPLACE_CHAR;
      }
      */
      uint32 decodeSingle(int b) {
         if (b>=0)
            return (uint32) b;
         return REPLACE_CHAR;
      }

      /*
      protected char decodeDouble(int byte1, int byte2) {
      if (((byte1 < 0) || (byte1 > index1.length))
      || ((byte2 < start) || (byte2 > end)))
      return REPLACE_CHAR;

      int n = (index1[byte1] & 0xf) * (end - start + 1) + (byte2 - start);
      return index2[index1[byte1] >> 4].charAt(n);
      }
      */
      uint32 decodeDouble(int byte1, int byte2) {
         if (((byte1 <0 ) || (byte1 > decoderTable1->len))
            || ((byte2 < start) || (byte2 > end)))
            return REPLACE_CHAR;

         int n = ( getUint16(decoderTable1, byte1)& 0xf) * (end - start + 1) + (byte2 - start);
         Table *charTable = (getTable(decoderTable2, getUint16(decoderTable1, byte1) >> 4));
         return getUTF16Char( (uint16*)charTable->table, n);
      }

      int encodeSingle(uint32 inputChar) {
         if (inputChar < 0x80)
            return (byte)inputChar;
         else
            return REPLACE_CHAR;
      }

      int encodeDouble(uint32 ch) {
         int offset = getUint16(encoderTable1,((ch & 0xff00) >> 8 )) << 8;
         Table *charTable = getTable(encoderTable2,offset >> 12);
         return getUTF16Char((uint16*)charTable->table, (offset & 0xfff) + (ch & 0xff));
      }

      virtual bool get( uint32 &chr )
      {
         fputs("\nGet processing, convert to Unicode\n", stderr);
         fflush(stderr);



         m_parseStatus = true;

         if( popBuffer( chr ) )
            return true;

         // converting the character into an unicode.
         byte b1,b2;
         if ( m_stream->read( &b1, 1 ) != 1 )
            return false;

         chr = decodeSingle(b1);
         if (chr == REPLACE_CHAR)
         {
            b1 &= 0xff;
            if (m_stream->read( &b2, 1 ) != 1)
               return false;
            chr = decodeDouble(b1, b2);
         }

         if (chr == REPLACE_CHAR)
         {
            chr = (uint32) '?';
            m_parseStatus = false;
         }

         return true;
      }

      virtual bool put( uint32 chr )
      {
         fputs("\nPut processing, convert from Unicode\n", stderr);
         fflush(stderr);

         m_parseStatus = true;
         int b = encodeSingle(chr);
         byte bPtr[2];
         if ( b != REPLACE_CHAR )
         {
            bPtr[0] = (byte)b;
            return (m_stream->write( bPtr, 1 ) == 1);
         }
         int ncode = encodeDouble(chr);
         if (ncode == 0 || chr == 0)
         {
            m_parseStatus = false;
            return false;
         }
         bPtr[0] = (byte)((ncode & 0xff00) >> 8);
         bPtr[1] = (byte)(ncode);
         return ( m_stream->write(bPtr, 2) == 2 );
      }

      virtual const String encoding() const { return "gbk"; }
      virtual FalconData *clone() const
      {
            return new TranscoderGBK( *this );
      }
};

/** Base class for codepage like transcoders. */
class TranscoderISO_CP: public Transcoder
{
protected:
   TranscoderISO_CP( Stream *s, bool bOwn=false ):
       Transcoder( s, bOwn )
       {}

    TranscoderISO_CP( const TranscoderISO_CP &other );

       uint16 *m_directTable;
       uint32 m_dirTabSize;
       CP_ISO_UINT_TABLE *m_reverseTable;
       uint32 m_revTabSize;
public:


   virtual bool get( uint32 &chr );
   virtual bool put( uint32 chr );
};

class TranscoderCP1252:public TranscoderISO_CP
{
public:
   TranscoderCP1252( const TranscoderCP1252 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderCP1252( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "cp1252"; }
     virtual FalconData *clone() const;
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
     virtual FalconData *clone() const;
};

class TranscoderISO8859_2: public TranscoderISO_CP
{
public:
   TranscoderISO8859_2( const TranscoderISO8859_2 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_2( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-2"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_3: public TranscoderISO_CP
{
public:
   TranscoderISO8859_3( const TranscoderISO8859_3 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_3( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-3"; }
     virtual FalconData *clone() const;
};


class TranscoderISO8859_4: public TranscoderISO_CP
{
public:
   TranscoderISO8859_4( const TranscoderISO8859_4 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_4( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-4"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_5: public TranscoderISO_CP
{
public:
   TranscoderISO8859_5( const TranscoderISO8859_5 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_5( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-5"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_6: public TranscoderISO_CP
{
public:
   TranscoderISO8859_6( const TranscoderISO8859_6 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_6( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-6"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_7: public TranscoderISO_CP
{
public:
   TranscoderISO8859_7( const TranscoderISO8859_7 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_7( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-7"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_8: public TranscoderISO_CP
{
public:
   TranscoderISO8859_8( const TranscoderISO8859_8 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_8( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-8"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_9: public TranscoderISO_CP
{
public:
   TranscoderISO8859_9( const TranscoderISO8859_9 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_9( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-9"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_10: public TranscoderISO_CP
{
public:
   TranscoderISO8859_10( const TranscoderISO8859_10 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_10( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-10"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_11: public TranscoderISO_CP
{
public:
   TranscoderISO8859_11( const TranscoderISO8859_11 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_11( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-11"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_13: public TranscoderISO_CP
{
public:
   TranscoderISO8859_13( const TranscoderISO8859_13 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_13( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-13"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_14: public TranscoderISO_CP
{
public:
   TranscoderISO8859_14( const TranscoderISO8859_14 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_14( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-14"; }
     virtual FalconData *clone() const;
};

class TranscoderISO8859_15: public TranscoderISO_CP
{
public:
   TranscoderISO8859_15( const TranscoderISO8859_15 &other ):
     TranscoderISO_CP( other )
     {}

     TranscoderISO8859_15( Stream *s, bool bOwn=false );
     virtual const String encoding() const { return "iso8859-15"; }
     virtual FalconData *clone() const;
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

FalconData *TranscoderByte::clone() const
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

FalconData *TranscoderEOL::clone() const
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

FalconData *TranscoderUTF8::clone() const
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

FalconData *TranscoderUTF16::clone() const
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
   m_revTabSize( other.m_revTabSize )
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
   if ( chr < 0xA0 )
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

FalconData *TranscoderCP1252::clone() const
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

FalconData *TranscoderISO8859_1::clone() const
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

FalconData *TranscoderISO8859_2::clone() const
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

FalconData *TranscoderISO8859_3::clone() const
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

FalconData *TranscoderISO8859_4::clone() const
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

FalconData *TranscoderISO8859_5::clone() const
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

FalconData *TranscoderISO8859_6::clone() const
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

FalconData *TranscoderISO8859_7::clone() const
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

FalconData *TranscoderISO8859_8::clone() const
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

FalconData *TranscoderISO8859_9::clone() const
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

FalconData *TranscoderISO8859_10::clone() const
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

FalconData *TranscoderISO8859_11::clone() const
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

FalconData *TranscoderISO8859_13::clone() const
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

FalconData *TranscoderISO8859_14::clone() const
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

FalconData *TranscoderISO8859_15::clone() const
{
   return new TranscoderISO8859_15( *this );
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
      return new TranscoderGBK(stream, own);
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

bool GetSystemEncoding( String &encoding )
{
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
         pos = encoding.find( "UTF-" );
         if ( pos == csh::npos )
            pos = encoding.find( "utf-" );

         if ( pos != csh::npos && pos < encoding.length() - 4 )
         {
            encoding = encoding.subString( pos + 4 );
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
   if ( encoding.length() > 2 )
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
}



}

/* end of transcoding.cpp */

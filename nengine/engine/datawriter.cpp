/*
   FALCON - The Falcon Programming Language.
   FILE: datawriter.cpp

   Data-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 21:30:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/endianity.h>
#include <falcon/stream.h>
#include <falcon/datawriter.h>

#include <falcon/ioerror.h>

#include <string.h>

namespace Falcon {

DataWriter::DataWriter( Stream* stream, t_endianity endian, bool bOwn ):
   Writer( stream, bOwn )
{
   setEndianity( endian );
}


DataWriter::DataWriter( t_endianity endian )
{
   setEndianity( endian );
}

DataWriter::~DataWriter()
{
}


void DataWriter::setEndianity( t_endianity endian )
{
   
   #if FALCON_LITTLE_ENDIAN == 1
   if( endian == e_sameEndian )
   {
      m_endianity = e_LE;
   }
   else if( endian == e_reverseEndian )
   {
      m_endianity = e_BE;
   }
   else
   {
      m_endianity = endian;
   }

   m_bIsSameEndianity = endian == e_LE;

   #else
   if( endian == e_sameEndian )
   {
      m_endianity = e_BE;
   }
   else if( endian == e_reverseEndian )
   {
      m_endianity = e_LE;
   }
   else
   {
      m_endianity = endian;
   }

   m_bIsSameEndianity = endian == e_BE;
   #endif
}


bool DataWriter::write(bool value)
{
   if( m_bufPos == m_bufSize ) flush();
   m_buffer[m_bufPos] = value ? 1:0;
   m_bufPos++;
   
   return true;
}

  
bool DataWriter::write(char value)
{
   if( m_bufPos == m_bufSize ) flush();
   m_buffer[m_bufPos] = (byte) value;
   m_bufPos++;

   return true;
}


 
bool DataWriter::write(byte value)
{
   if( m_bufPos == m_bufSize ) flush();
   m_buffer[m_bufPos] = value;
   m_bufPos++;

   return true;
}


bool DataWriter::write( uint16 value )
{
   byte* addr = (byte*) &value;
   byte locBuf[2];
   if( m_bIsSameEndianity )
   {
      locBuf[0] = addr[0];
      locBuf[1] = addr[1];
   }
   else
   {
      locBuf[1] = addr[0];
      locBuf[0] = addr[1];
   }

   return write( locBuf, 2 );
}

bool DataWriter::write( uint32 value )
{
   byte* addr = (byte*) &value;
   byte locBuf[4];
   if( m_bIsSameEndianity )
   {
      locBuf[0] = addr[0];
      locBuf[1] = addr[1];
      locBuf[2] = addr[2];
      locBuf[3] = addr[3];
   }
   else
   {
      locBuf[3] = addr[0];
      locBuf[2] = addr[1];
      locBuf[1] = addr[2];
      locBuf[0] = addr[3];
   }

   return write( locBuf, 4 );
}


bool DataWriter::write( uint64 value )
{
   byte* addr = (byte*) &value;
   byte locBuf[8];
   if( m_bIsSameEndianity )
   {
      locBuf[0] = addr[0];
      locBuf[1] = addr[1];
      locBuf[2] = addr[2];
      locBuf[3] = addr[3];
      locBuf[4] = addr[4];
      locBuf[5] = addr[5];
      locBuf[6] = addr[6];
      locBuf[7] = addr[7];
   }
   else
   {
      locBuf[7] = addr[0];
      locBuf[6] = addr[1];
      locBuf[5] = addr[2];
      locBuf[4] = addr[3];
      locBuf[3] = addr[4];
      locBuf[2] = addr[5];
      locBuf[1] = addr[6];
      locBuf[0] = addr[7];
   }

   return write( locBuf, 8 );
}


bool DataWriter::write( float value )
{
   byte* addr = (byte*) &value;
   byte locBuf[4];
   if( m_bIsSameEndianity )
   {
      locBuf[0] = addr[0];
      locBuf[1] = addr[1];
      locBuf[2] = addr[2];
      locBuf[3] = addr[3];
   }
   else
   {
      locBuf[3] = addr[0];
      locBuf[2] = addr[1];
      locBuf[1] = addr[2];
      locBuf[0] = addr[3];
   }

   return write( locBuf, 4 );
}


bool DataWriter::write( double value )
{
   byte* addr = (byte*) &value;
   byte locBuf[8];
   if( m_bIsSameEndianity )
   {
      locBuf[0] = addr[0];
      locBuf[1] = addr[1];
      locBuf[2] = addr[2];
      locBuf[3] = addr[3];
      locBuf[4] = addr[4];
      locBuf[5] = addr[5];
      locBuf[6] = addr[6];
      locBuf[7] = addr[7];
   }
   else
   {
      locBuf[7] = addr[0];
      locBuf[6] = addr[1];
      locBuf[5] = addr[2];
      locBuf[4] = addr[3];
      locBuf[3] = addr[4];
      locBuf[2] = addr[5];
      locBuf[1] = addr[6];
      locBuf[0] = addr[7];
   }

   return write( locBuf, 8 );
}


bool DataWriter::write( const String& tgt )
{
   byte nCharCount=tgt.manipulator()->charSize();
   length_t size = tgt.size();

   write( nCharCount );
   write( size );

   if( size > 0 )
   {
      if( m_bIsSameEndianity || nCharCount == 1 )
      {
         return write( tgt.getRawStorage(), size );
      }
      else
      {
         String temp;
         temp.bufferize(tgt);
         if( nCharCount == 2 )
         {
            REArray_16Bit( temp.getRawStorage(), size/2 );
         }
         else if( nCharCount == 4 )
         {
            REArray_32Bit( temp.getRawStorage(), size/4 );
         }
         else
         {
            fassert( 0 );
         }

         return write( tgt.getRawStorage(), size );
      }
   }

   return true;
}

}

/* end of datawriter.cpp */

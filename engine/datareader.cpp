/*
   FALCON - The Falcon Programming Language.
   FILE: datareader.cpp

   Data-oriented stream reader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 17 Mar 2011 11:36:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/endianity.h>
#include <falcon/stream.h>
#include <falcon/datareader.h>
#include <falcon/errors/ioerror.h>

#include <string.h>

namespace Falcon {

DataReader::DataReader( Stream* stream, t_endianity endian, bool bOwn ):
   Reader( stream, bOwn ),
   m_gcMark( 0 )
{
   setEndianity( endian );
}


DataReader::DataReader( t_endianity endian ):
   m_gcMark( 0 )
{
   setEndianity( endian );
}


DataReader::DataReader( const DataReader& other ):
   Reader( m_stream ? m_stream->clone() : 0 , true ),
   m_gcMark( 0 )
{
   setEndianity( other.m_endianity );
}


DataReader::~DataReader()
{
}


void DataReader::setEndianity( t_endianity endian )
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


bool DataReader::read(bool &value)
{
   // ensure will throw if requested to
   if( ! ensure(1) ) return false;
   
   value = m_buffer[m_bufPos] == 1;
   m_bufPos++;
   return true;
}

  
bool DataReader::read(char &value)
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(byte)) ) return false;
   value = (char) m_buffer[m_bufPos];
   m_bufPos += sizeof(byte);
   return true;
}


 
bool DataReader::read(byte &value)
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(byte)) ) return false;
   value = m_buffer[m_bufPos];
   m_bufPos += sizeof(byte);
   return true;
}


bool DataReader::read( int16 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(int16)) ) return false;
   value = (int16)( m_bIsSameEndianity ? getUInt16( m_buffer + m_bufPos ) : getUInt16Reverse( m_buffer + m_bufPos ));
   m_bufPos += sizeof(int16);
   return true;
}


bool DataReader::read( uint16 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(int16)) ) return false;
   value = m_bIsSameEndianity ? getUInt16( m_buffer + m_bufPos ) : getUInt16Reverse( m_buffer + m_bufPos );
   m_bufPos += sizeof(int16);
   return true;
}


bool DataReader::read( int32 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(int32)) ) return false;
   value = (int32)( m_bIsSameEndianity ? getUInt32( m_buffer + m_bufPos ) : getUInt32Reverse( m_buffer + m_bufPos ));
   m_bufPos += sizeof(int32);
   return true;
}


bool DataReader::read( uint32 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(uint32)) ) return false;
   value = ( m_bIsSameEndianity ? getUInt32( m_buffer + m_bufPos ) : getUInt32Reverse( m_buffer + m_bufPos ));
   m_bufPos += sizeof(uint32);
   return true;
}


bool DataReader::read( int64 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(int64)) ) return false;
   value = (int64)( m_bIsSameEndianity ? getUInt64( m_buffer + m_bufPos ) : getUInt64Reverse( m_buffer + m_bufPos ));
   m_bufPos += sizeof(int64);
   return true;
}


bool DataReader::read( uint64 &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(uint64)) ) return false;
   value = (uint64)( m_bIsSameEndianity ? getUInt64( m_buffer + m_bufPos ) : getUInt64Reverse( m_buffer + m_bufPos ));
   m_bufPos += sizeof(uint64);
   return true;
}


bool DataReader::read( float &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(float)) ) return false;
   value = m_bIsSameEndianity ? getFloat32( m_buffer + m_bufPos ) : getFloat32Reverse( m_buffer + m_bufPos );
   m_bufPos += sizeof(float);
   return true;
}


bool DataReader::read( double &value )
{
   // ensure will throw if requested to
   if( ! ensure(sizeof(double)) ) return false;
   value = m_bIsSameEndianity ? getFloat64( m_buffer + m_bufPos ) : getFloat64Reverse( m_buffer + m_bufPos );
   m_bufPos += sizeof(double);
   return true;
}



bool DataReader::read( byte* buffer, length_t size )
{
   // ensure will throw if requested to
   if( ! ensure(size) ) return false;
   memcpy( buffer, m_buffer + m_bufPos, size );
   m_bufPos += size;
   return true;
}


bool DataReader::read( String& tgt )
{
   byte nCharCount=0;
   length_t size = (length_t)-1;

   try
   {
      if( ! read(nCharCount) ) return false;
      
      switch( nCharCount )
      {
         case 1: tgt.manipulator(&csh::handler_buffer); break;
         case 2: tgt.manipulator(&csh::handler_buffer16); break;
         case 4: tgt.manipulator(&csh::handler_buffer32); break;
         default:
            if( m_stream->shouldThrow() )
            {
               throw new IOError( ErrorParam(e_deser, __LINE__, __FILE__ ));
            }
            return false;
      }

      if( !read(size) ) return false;
      if( size % nCharCount != 0 )
      {
         throw new IOError( ErrorParam(e_deser, __LINE__, __FILE__ ));
      }

      if( size > 0 )
      {
         tgt.reserve( size );
         if( !read( tgt.getRawStorage(), size ) ) return false;

         // eventually reverse the endianity
         if( ! m_bIsSameEndianity )
         {
            if( nCharCount == 2 )
            {
               REArray_16Bit( tgt.getRawStorage(), size/2 );
            }
            else if( nCharCount == 4 )
            {
               REArray_32Bit( tgt.getRawStorage(), size/4 );
            }
         }
      }

      tgt.size(size);
   }
   catch( IOError* e )
   {
      e->extraDescription("String");
      throw;
   }
   
   return true;
}

}

/* end of datareader.cpp */

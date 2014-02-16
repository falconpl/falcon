/*
   FALCON - The Falcon Programming Language.
   FILE: bytebuf.cpp

   Buffering extensions
   Bit-perfect buffer class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Jul 2013 13:22:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   Licensed under the Falcon Programming Language License,
   Version 1.1 (the "License"); you may not use this file
   except in compliance with the License. You may obtain
   a copy of the License at

   http://www.falconpl.org/?page_id=license_1_1

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on
   an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied. See the License for the
   specific language governing permissions and limitations
   under the License.

*/

#include <falcon/types.h>
#include <falcon/fassert.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>

#include <cstring>

#include "bitbuf_mod.h"

namespace Falcon {
namespace Ext {

void BitBuf::init()
{
   m_first = 0;
   m_last = 0;

   m_readpos = 0;
   m_writepos = 0;

   m_read_endianity = m_write_endianity = e_endian_little;
   m_sys_endianity = e_endian_little;

   m_size = 0;
}


BitBuf::t_endianity BitBuf::sysEndianity()
{
   // Check the system endianity
   uint16 test = 0xFFFE;
   byte* btest = (byte*) &test;
   return  btest[0] == 0xFF ? e_endian_little : e_endian_big;
}


void BitBuf::toString( String& target, char_t chrOn, char_t chrOff )
{
   target.size(0);
   target.reserve(m_size);
   uint32 currentRpos = rpos();
   rpos(0);
   bool bit;
   while( readBit(bit) )
   {
      target.append( (char_t) (bit ? chrOn : chrOff) );
   }
   rpos( currentRpos );
}


uint32 BitBuf::storageSize() const
{
   uint32 count = m_size/8;
   if( count * 8 != m_size ) count ++;
   return count;
}


void BitBuf::clear()
{
   Chunk* chunk = m_first;
   while( chunk != 0 )
   {
      Chunk* old = chunk;
      chunk = chunk->m_next;
      delete[] reinterpret_cast<byte*>(old);
   }

   m_first = m_last = 0;
   m_size = 0;
   m_readpos = 0;
   m_writepos = 0;
}


byte* BitBuf::consolidate()
{
   if( m_first != m_last )
   {
      uint32 bytesSize = storageSize();

      Chunk* newChunk = allocChunk( bytesSize );
      newChunk->m_next = 0;
      newChunk->m_basePos = 0;

      // fully used chunks will always have an even number of occupied bits.
      Chunk* chunk = m_first;
      uint32 pos = 0;
      uint32 count = 0;
      // keep the last block
      while( chunk != 0 )
      {
         count = chunk->m_usedBits/8;
         memcpy(newChunk->m_memory + pos, chunk->m_memory, count);
         pos += count;
         Chunk* oldChunk = chunk;
         chunk = chunk->m_next;
         if( chunk != 0 )
         {
            delete[] reinterpret_cast<byte*>(oldChunk);
         }
      }

      // copy the last bits in the last chunk.
      uint32 lastBase = count * 8;
      uint32 restBits = m_last->m_usedBits - lastBase;
      if( restBits != 0 )
      {
         newChunk->m_memory[pos] = m_last->m_memory[m_last->m_usedBits/8];
      }
      newChunk->m_sizeBytes = bytesSize;
      newChunk->m_usedBits = pos * 8 + restBits;
      fassert( m_size == newChunk->m_usedBits );
      m_size = newChunk->m_usedBits;  // should be the same...

      // kill the last block as well
      delete[] reinterpret_cast<byte*>(m_last);
      m_first = m_last = newChunk;

      return (byte*) m_first->m_memory;
   }

   if( m_first != 0 )
   {
      return (byte*) m_first->m_memory;
   }

   return 0;
}


void BitBuf::write16_direct( uint16 number )
{
   writeBytes( reinterpret_cast<byte*>(&number), 2 );
}


void BitBuf::write32_direct( uint32 number )
{
   writeBytes( reinterpret_cast<byte*>(&number), 4 );
}


void BitBuf::write64_direct( uint64 number )
{
   writeBytes( reinterpret_cast<byte*>(&number), 8 );
}


void BitBuf::write16_reverse( uint16 number )
{
   byte* bits = (byte*) &number;
   write8( bits[1] );
   write8( bits[0] );
}


void BitBuf::write32_reverse( uint32 number )
{
   byte* bits = (byte*) &number;
   write8( bits[3] );
   write8( bits[2] );
   write8( bits[1] );
   write8( bits[0] );
}

void BitBuf::write64_reverse( uint64 number )
{
   byte* bits = (byte*) &number;
   write8( bits[7] );
   write8( bits[6] );
   write8( bits[5] );
   write8( bits[4] );
   write8( bits[3] );
   write8( bits[2] );
   write8( bits[1] );
   write8( bits[0] );
}

void BitBuf::writeBytes( const byte* memory, uint32 count )
{
   uint32 bwp = m_writepos / 8;
   uint32 srest = m_writepos - (bwp * 8);

   if( count == 0 )
   {
      return;
   }

   if ( srest == 0 )
   {
      // even start, so...
      byte* target = reserveEvenBytes( count );
      memcpy( target, memory, count );
      m_writepos += count * 8;
      if( m_writepos > m_size )
      {
         m_last->m_usedBits = m_writepos - m_last->m_basePos;
         m_size = m_writepos;
      }
   }
   else {
      // nope, we must write unaligned bits.
      uint32 shift = 7;
      const byte* mempos = memory;
      uint32 bitCount = 8 * count;
      for( uint32 i = 0; i < bitCount; ++i )
      {
         bool bit = (*mempos & (0x1) << shift) != 0;
         writeBit( bit );

         if( shift == 0 )
         {
            shift = 7;
            mempos++;
         }
         else {
            shift--;
         }
      }
   }
}


byte* BitBuf::reserveEvenBytes( uint32 count )
{
   // we assert that the current write position is even.
   fassert( m_writepos % 8 == 0 );
   // also, any write pointer move operation should have consolidated, so...
   fassert( m_last == 0 || m_last->m_basePos <= m_writepos );


   // then, we can decide whether we can use the remaining chunk...
   uint32 woffsetBytes = m_last == 0 ? 0 : (m_writepos - m_last->m_basePos) / 8;
   if( m_last != 0 && (m_last->m_sizeBytes - woffsetBytes) >= count )
   {
      // good, we can return the current chunk data.
      return m_last->m_memory + (m_writepos/8);
   }
   else
   {
      // get a chunk large enough.
      Chunk* newChunk = allocChunk( count );
      newChunk->m_basePos = m_writepos;

      if ( m_last != 0 )
      {
         // we're better off truncating the chunk
         m_last->m_sizeBytes = woffsetBytes;
         m_last->m_next = newChunk;
      }
      else {
         m_first = newChunk;
      }

      m_last = newChunk;

      return newChunk->m_memory;
   }
}

void BitBuf::writeBits( const byte* memory, uint32 count, uint32 start )
{
   if( start % 8 == 0 )
   {
      memory = memory + (start/8);

      uint32 bcount = count/8;
      if( bcount > 0 )
      {
         writeBytes( memory, bcount );
      }

      uint32 rest = count - bcount*8;
      uint8 b = memory[bcount];
      if( rest > 0 )
      {
         writeSomeBits( b, rest );
      }
   }
   else {
      uint32 posStart = start / 8;
      memory += posStart;
      uint32 rest = start - (posStart*8);

      while( count > 0 )
      {
         bool bitOn = (*memory & (0x80 >> rest) ) != 0;
         writeBit(bitOn);

         if( rest == 7 )
         {
            rest = 0;
            ++memory;
         }
         else {
            ++rest;
         }
         --count;
      }
   }
}


// writes less bits than the given count
void BitBuf::writeSomeBits( byte b, uint32 count )
{
   fassert( count <= 8 );

   while( count > 0 )
   {
      writeBit( (b & 0x80) ? true : false );
      b <<= 1;
      count--;
   }
}

void BitBuf::writeBit( bool bit )
{
   if( m_last == 0 ||
            (m_last->m_basePos + (m_last->m_sizeBytes * 8) == m_writepos ) )
   {
      appendNewChunk();
   }

   uint32 displacement = m_writepos - m_last->m_basePos;

   uint32 bytepos = displacement/8;
   uint32 bitpos = displacement - (bytepos*8);
   byte mask;
   if( bit ) {
      mask = 0x01 << (7-bitpos);
      m_last->m_memory[bytepos] |= mask;
   }
   else {
      mask = (0xfe << (7-bitpos)) | (0x7f >> bitpos);
      m_last->m_memory[bytepos] &= mask;
   }

   m_writepos++;
   displacement++;

   if( displacement > m_last->m_usedBits )
   {
      m_last->m_usedBits = displacement;
   }

   if( m_writepos > m_size )
   {
      m_size = m_writepos;
   }
}


void BitBuf::appendNewChunk()
{
   // ensure we call this function at even points only.
   fassert( m_last == 0 || m_last->m_usedBits % 8 == 0);

   uint32 basePos;
   if( m_first == 0 )
   {
      m_first = m_last = allocChunk( DEFAULT_CHUNK_SIZE );
      basePos = 0;
   }
   else {
      basePos = m_last->m_basePos + m_last->m_usedBits;
      m_last->m_next = allocChunk( DEFAULT_CHUNK_SIZE );
      m_last = m_last->m_next;
   }

   m_last->m_basePos = basePos;
}


BitBuf::Chunk* BitBuf::allocChunk( uint32 byteSize )
{
   if( byteSize < DEFAULT_CHUNK_SIZE )
   {
      byteSize = DEFAULT_CHUNK_SIZE;
   }

   Chunk* chunk = reinterpret_cast<Chunk*>( new byte[ byteSize + sizeof(Chunk) ] );
   chunk->m_basePos = 0;
   chunk->m_next = 0;
   chunk->m_sizeBytes = byteSize;
   chunk->m_usedBits = 0;

   return chunk;
}


void BitBuf::write8( uint8 byte )
{
   if( m_writepos % 8 == 0 )
   {
      if( m_last == 0 ||
               m_writepos +8 >= (m_last->m_basePos + (m_last->m_sizeBytes * 8) ) )
      {
         appendNewChunk();
      }

      uint32 displacement = ( m_writepos - m_last->m_basePos ) /8;
      m_last->m_memory[displacement] = byte;
      m_writepos += 8;
   }
   else {
      writeSomeBits( byte, 8 );
   }
}


void BitBuf::write16( uint16 number )
{
   if( writeStraightBitOrder() )
   {
      write16_direct(number);
   }
   else {
      write16_reverse(number);
   }
}


void BitBuf::write32( uint32 number )
{
   if( writeStraightBitOrder() )
   {
      write32_direct(number);
   }
   else {
      write32_reverse(number);
   }
}


void BitBuf::write64( uint64 number )
{
   if( writeStraightBitOrder() )
   {
      write64_direct(number);
   }
   else {
      write64_reverse(number);
   }
}


void BitBuf::write( const BitBuf& other )
{
   clear();

   other.lock();
   uint32 bytesSize = other.storageSize();
   if( bytesSize != 0 )
   {
      Chunk* newChunk = (Chunk*) new byte[bytesSize];
      newChunk->m_next = 0;
      newChunk->m_basePos = 0;

      // fully used chunks will always have an even number of occupied bits.
      Chunk* chunk = other.m_first;
      uint32 pos = 0;
      uint32 count = 0;
      while( chunk != 0 )
      {
         count = chunk->m_usedBits/8;
         memcpy(newChunk->m_memory + pos, chunk->m_memory, count);
         pos += count;
         chunk = chunk->m_next;
      }

      // copy the last bits in the last chunk.
      uint32 lastBase = count * 8;
      uint32 restBits = m_last->m_usedBits - lastBase;
      if( restBits != 0 )
      {
         newChunk->m_memory[pos] = m_last->m_memory[lastBase];
      }
      newChunk->m_sizeBytes = bytesSize;
      newChunk->m_usedBits = pos * 8 + restBits;
      m_size = newChunk->m_usedBits;  // should be the same...

      m_first = m_last = newChunk;
   }

   other.unlock();
}


//==================================================================
// read
//

uint32 BitBuf::readBytes( byte* memory, uint32 count )
{
   lock();
   count = readBytes_internal( memory, count );
   unlock();

   return count;
}


uint32 BitBuf::readBytes_internal( byte* memory, uint32 count )
{
   if( count * 8  > readable() )
   {
      count = readable() / 8;
   }

   if (count == 0 )
   {
      return 0;
   }

   if( m_readpos % 8 != 0 )
   {
      uint32 ncount = readBits_internal( memory, count *8 ) / 8 ;
      return ncount;
   }

   byte* src = consolidate();
   memcpy( memory, src + m_readpos/8, count );
   m_readpos += count*8;

   return count;
}

uint32 BitBuf::readBits( byte* memory, uint32 count )
{
   lock();
   count = readBits_internal( memory, count );
   unlock();

   return count;
}


uint32 BitBuf::readBits_internal( byte* memory, uint32 count )
{   
   if( count > readable() )
   {
      count = readable();
   }

   if( count == 0 )
   {
      return 0;
   }

   if( count % 8 == 0 && m_readpos % 8 == 0)
   {
      return readBytes_internal(memory, count/8) * 8;
   }

   byte* src = consolidate();
   uint32 pos = m_readpos/8;
   uint32 rest = m_readpos - (pos*8);
   src += pos;

   uint32 tbr = count;
   if( rest == 0 )
   {
      uint32 readCount = count / 8;
      memcpy( memory, src, readCount );
      tbr = count - readCount*8;
      src += readCount;
      memory += readCount;
   }

   uint32 srcBitPos = rest;
   rest = 0;

   while(tbr > 0)
   {
      if( rest == 0 )
      {
         *memory = 0;
      }

      byte mask = *src & (0x1 << (7-srcBitPos));
      if( mask != 0 )
      {
         *memory |= 1 << (7-rest);
      }
      else {
         //suppose memory is initialized to 0
         //memory &= 0xFE << (7-rest);
      }

      srcBitPos++;
      if( srcBitPos == 8 ) {
         src++;
         srcBitPos = 0;
      }

      rest++;
      if( rest == 8 )
      {
         memory++;         
         rest = 0;
      }
      --tbr;
   }

   m_readpos += count;

   return count;
}

bool BitBuf::read16( uint16& number )
{
   if( readStraightBitOrder() )
   {
      return read16_direct(number);
   }
   else {
      return read16_reverse(number);
   }
}


bool BitBuf::read32( uint32& number )
{
   if( readStraightBitOrder() )
   {
      return read32_direct(number);
   }
   else {
      return read32_reverse(number);
   }
}


bool BitBuf::read64( uint64& number )
{
   if( readStraightBitOrder() )
   {
      return read64_direct(number);
   }
   else {
      return read64_reverse(number);
   }
}


bool BitBuf::readBit( bool& bit )
{
   lock();
   if( readable() == 0 )
   {
      unlock();
      return false;
   }

   byte* memory = consolidate();
   uint32 pos = m_readpos/8;
   uint32 rest = m_readpos - (pos*8);
   bit = (memory[pos] & (0x1 << (7-rest))) != 0;
   m_readpos++;
   unlock();

   return true;
}


bool BitBuf::read8( uint8 &bt )
{
   if( readable() < 8 )
   {
      return false;
   }

   bt = 0;
   readBytes( reinterpret_cast<byte*>(&bt), 1 );
   return true;
}


bool BitBuf::read16_direct( uint16& number )
{
   if( readable() < 16 )
   {
      return false;
   }
   number = 0;
   readBytes( reinterpret_cast<byte*>(&number), 2 );
   return true;
}

bool BitBuf::read32_direct( uint32& number )
{
   if( readable() < 32 )
   {
      return false;
   }
   number = 0;
   readBytes( reinterpret_cast<byte*>(&number), 4 );
   return true;
}

bool BitBuf::read64_direct( uint64& number )
{
   if( readable() < 64 )
   {
      return false;
   }
   number = 0;
   readBytes( reinterpret_cast<byte*>(&number), 8 );
   return true;
}

bool BitBuf::read16_reverse( uint16& number )
{
   if( readable() < 16 )
   {
      return false;
   }
   uint16 temp = 0;
   byte* ptemp = reinterpret_cast<byte*>(&temp);
   readBytes( ptemp, 2 );

   byte* pnumber = reinterpret_cast<byte*>(&number);
   // readBytes might change ptemp
   ptemp = reinterpret_cast<byte*>(&temp);
   pnumber[0] = ptemp[1];
   pnumber[1] = ptemp[0];

   return true;
}


bool BitBuf::read32_reverse( uint32& number )
{
   if( readable() < 32 )
   {
      return false;
   }
   uint32 temp = 0;
   byte* ptemp = reinterpret_cast<byte*>(&temp);
   readBytes( ptemp, 4 );

   byte* pnumber = reinterpret_cast<byte*>(&number);
   ptemp = reinterpret_cast<byte*>(&temp);
   pnumber[0] = ptemp[3];
   pnumber[1] = ptemp[2];
   pnumber[2] = ptemp[1];
   pnumber[3] = ptemp[0];

   return true;
}


bool BitBuf::read64_reverse( uint64& number )
{
   if( readable() < 64 )
   {
      return false;
   }
   uint64 temp = 0;
   byte* ptemp = reinterpret_cast<byte*>(&temp);
   readBytes( ptemp, 8 );

   byte* pnumber = reinterpret_cast<byte*>(&number);
   // readBytes might change ptemp
   ptemp = reinterpret_cast<byte*>(&temp);
   pnumber[0] = ptemp[7];
   pnumber[1] = ptemp[6];
   pnumber[2] = ptemp[5];
   pnumber[3] = ptemp[4];
   pnumber[4] = ptemp[3];
   pnumber[5] = ptemp[2];
   pnumber[6] = ptemp[1];
   pnumber[7] = ptemp[0];

   return true;
}


bool BitBuf::wpos(uint32 pos)
{
   lock();
   if( pos > m_size )
   {
      unlock();
      return false;
   }

   consolidate();
   m_writepos = pos;
   unlock();
   return true;
}


bool BitBuf::rpos(uint32 pos)
{
   lock();
   if( pos > m_size )
   {
      unlock();
      return false;
   }

   consolidate();
   m_readpos = pos;
   unlock();
   return true;
}

bool BitBuf::eof()
{
   lock();
   bool status = m_size == m_readpos;
   unlock();

   return status;
}


void BitBuf::store( DataWriter* target )
{
   byte* stored = 0;

   lock();

   byte rend = (byte) m_read_endianity;
   byte wend = (byte) m_write_endianity;
   byte* memory = consolidate();

   if( memory == 0 )
   {
      unlock();

      target->write( rend );
      target->write( wend );
      target->write(0);
      return;
   }

   uint32 usedBits = (uint32) m_last->m_usedBits;
   uint32 size = storageSize();
   stored = new byte[size];
   memcpy( stored, memory, size );
   unlock();

   try {
      target->write( rend );
      target->write( wend );
      target->write( usedBits );
      target->writeRaw( stored, size );
      delete[] stored;
   }
   catch( ... )
   {
      delete[] stored;
      throw;
   }
}


void BitBuf::restore( DataReader* source )
{
   byte rend;
   byte wend;
   source->read(rend);
   source->read(wend);

   uint64 bitCount = 0;
   source->read(bitCount);

   Chunk* chunk = 0;
   if( bitCount != 0 )
   {
      uint64 byteCount = bitCount/8;
      if( byteCount*8 != bitCount ) byteCount++;
      chunk = allocChunk((uint32)byteCount);
      try {
         source->read( chunk->m_memory, (uint32) byteCount );
      }
      catch( ... )
      {
         delete chunk;
         throw;
      }

      chunk->m_sizeBytes = (uint32) byteCount;
      chunk->m_usedBits = (uint32) bitCount;
      chunk->m_basePos = 0;
      chunk->m_next = 0;
   }

   lock();
   m_read_endianity = (t_endianity) rend;
   m_write_endianity = (t_endianity) wend;
   m_first = m_last = chunk;
   m_size = (uint32) bitCount;
   unlock();
}


void BitBuf::writeNumberBits( uint64 number, uint32 bits )
{
   while( bits > 0 )
   {
      --bits;
      writeBit( ((number >> bits) & 0x1) != 0 );
   }
}


bool BitBuf::readNumberBits( uint64& number, uint32 bits )
{
   if( readable() < bits )
   {
      unlock();
      return false;
   }

   number = 0;
   while( bits > 0 )
   {
      bool bit = false;
      readBit( bit );
      number <<= 1;
      number |= bit ? 1 : 0;
      --bits;
   }

   return true;
}

}} // Namespace Falcon::Ext

/* end of bitbuf.cpp */

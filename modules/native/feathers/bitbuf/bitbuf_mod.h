/*
   FALCON - The Falcon Programming Language.
   FILE: bytebuf.h

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

#ifndef FALCON_BITBUF_MOD_H
#define FALCON_BITBUF_MOD_H

#include <falcon/types.h>
#include <falcon/mt.h>

namespace Falcon {
class DataReader;
class DataWriter;

namespace Ext {


class BitBuf
{
public:
    typedef uint8 VALTYPE; // <- this can be used to specify the underlying integer type, which should be *unsigned*. (uint8...uint64)                            // sizeof(NUMTYPE) *MUST* be >= sizeof(VALTYPE)
    static const VALTYPE VAL_ONE = 1; // used in shift operations (1 << X). must be always 1, and explicitly of type VALTYPE.
    static const VALTYPE VAL_ALLBITS = 0xFF; // used in shift operations (VAL_ALLBITS >> X). must set all bits of VALTYPE.

    enum { VALBITS = sizeof(VALTYPE) * 8 };
    enum { DEFAULT_CHUNK_SIZE = 128 };

    typedef enum {
       e_endian_little,
       e_endian_big,
       e_endian_same,
       e_endian_reverse
    }
    t_endianity;

    BitBuf()
    {
       init();
    }

    BitBuf( const BitBuf& other )
    {
       init();
       write(other);
    }

    ~BitBuf()
    {
       clear();
    }

    /** Put all the data in a single segment and return the consolidated memory.
     * \return a sequential memory where all the bits are written, or 0 if the buffer is empty.
     */
    byte* consolidate();

    /** Locks this object for parallel access. */
    void lock() const { m_mtx.lock(); }

    /** Unlocks this object for parallel access. */
    void unlock() const { m_mtx.unlock(); }

    /** Clears the buffer */
    void clear();

    /** Returns the size in bits of the data stored in this buffer. */
    uint32 size() const { return m_size; }

    /** Count of minimum number of bytes required to store all the bits in this buffer */
    uint32 storageSize() const;

    void writeBit( bool bit );
    void write8( uint8 byte );
    void write16( uint16 number );
    void write32( uint32 number );
    void write64( uint64 number );

    void write( const BitBuf& other );

    void writeBytes( const byte* memory, uint32 count );
    void writeBits( const byte* memory, uint32 count, uint32 start = 0 );

    bool readBit( bool& bit );
    bool read8( uint8 &byte );
    bool read16( uint16& number );
    bool read32( uint32& number );
    bool read64( uint64& number );

    inline uint8 read8() { uint8 number = 0; read8(number); return number; }
    inline uint16 read16() { uint16 number = 0; read16(number); return number; }
    inline uint32 read32() { uint32 number = 0; read32(number); return number; }
    inline uint64 read64() { uint64 number = 0; read64(number); return number; }


    uint32 readBytes( byte* memory, uint32 count );
    uint32 readBits( byte* memory, uint32 count );

    bool eof();

    void store( DataWriter* target );
    void restore( DataReader* source );

    uint32 currentMark() const { return _gcMark; }
    void gcMark( uint32 mark ) { _gcMark = mark; }

    // amount of bits required to store an int number of a certain value
    static uint32 bits_req(uint64 n)
    {
        uint32 r = 0;
        while(n)
        {
            n >>= 1;
            ++r;
        }
        return r;
    }

    /** Returns the write position in the stream in bits */
    inline uint32 wpos() const { return (uint32)m_writepos; }

    /** Returns the read position in the stream in bits */
    inline uint32 rpos() const { return (uint32)m_readpos; }

    /** Number of bits that can still be read from the stream. */
    inline uint32 readable() const { return (m_size - m_readpos); }

    /** Sets the write pointer at the given position in the bit stream.
     * \param pos the position in bits from the start of the bit stream.
     *
     * The operation might consolidate the buffer.
     */
    bool wpos(uint32 pos);

    /** Sets the read pointer at the given position in the bit stream.
     * \param pos the position in bits from the start of the bit stream.
     *
     * The operation might consolidate the buffer.
     */
    bool rpos(uint32 pos);


    t_endianity writeEndianity() const { return m_write_endianity; }
    t_endianity readEndianity() const { return m_read_endianity; }
    void writeEndianity( t_endianity e ) { m_write_endianity = e; }
    void readEndianity( t_endianity e ) { m_read_endianity = e; }

    static t_endianity sysEndianity();

    void toString( String& target, char_t chrOn='1', char_t chrOff='0' );

    void writeNumberBits( uint64 number, uint32 bits );
    bool readNumberBits( uint64& number, uint32 bits );

private:

   typedef struct tag_CHUNK
   {
     uint32 m_sizeBytes;
     uint32 m_usedBits;
     uint32 m_basePos;
     tag_CHUNK* m_next;
     VALTYPE m_memory[1];
   }
   Chunk;

   Chunk* m_first;
   Chunk* m_last;

   uint32 m_readpos;
   uint32 m_writepos;

   t_endianity m_write_endianity;
   t_endianity m_read_endianity;
   t_endianity m_sys_endianity;

   uint32 m_size;

   mutable Mutex m_mtx;

   /** Initializes the buffer */
   void init();

    void write16_direct( uint16 number );
    void write32_direct( uint32 number );
    void write64_direct( uint64 number );
    void write16_reverse( uint16 number );
    void write32_reverse( uint32 number );
    void write64_reverse( uint64 number );

    bool read16_direct( uint16& number );
    bool read32_direct( uint32& number );
    bool read64_direct( uint64& number );
    bool read16_reverse( uint16& number );
    bool read32_reverse( uint32& number );
    bool read64_reverse( uint64& number );

    void writeNumberBits_little( uint64 number, int32 bits );
    bool readNumberBits_little( uint64& number, int32 bits );

    void writeNumberBits_big( uint64 number, int32 bits );
    bool readNumberBits_big( uint64& number, int32 bits );

    uint32 readBytes_internal( byte* memory, uint32 count );
    uint32 readBits_internal( byte* memory, uint32 count );

    // allocates an empty chunk of a given byte size.
    Chunk* allocChunk( uint32 byteSize );

    // Write a count of bits less than a byte.
    void writeSomeBits( byte b, uint32 count );

    // append a new default-sized chunk at the end of an 8-bit aligned write.
    void appendNewChunk();

    // Reserve a whole (8-bit alined) count of bytes
    byte* reserveEvenBytes( uint32 count );

    bool writeStraightBitOrder() {
       return m_write_endianity == m_sys_endianity || m_write_endianity == e_endian_same ;
    }

    bool readStraightBitOrder() {
       return m_read_endianity == m_sys_endianity || m_read_endianity == e_endian_same;
    }

    uint32 _gcMark;
};

}}  // Namespace Falcon::Ext

#endif

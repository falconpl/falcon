/*
   FALCON - The Falcon Programming Language.
   FILE: hash_ext.cpp

   Provides multiple hashing algorithms
   Interface extension functions
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Thu, 25 Mar 2010 02:46:10 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: The above AUTHOR

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

/** \file
   Provides multiple hashing algorithms
   Internal logic functions - implementation.
*/

#include <falcon/engine.h>
#include <falcon/class.h>
#include <falcon/autocstring.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <string.h>
#include "hash_mod.h"


namespace Falcon {
namespace Mod {

HashBase::HashBase( Class* cls ):
         _finalized(false),
         m_gcMark(0),
         m_handler(cls)
{
}

HashBase::HashBase( const HashBase& other ):
         _finalized( other._finalized ),
         m_gcMark( other.m_gcMark ),
         m_handler( other.m_handler )
{
}


HashBase::~HashBase()
{}

void HashBase::store(DataWriter* stream) const
{
   stream->write( _finalized );
}

void HashBase::restore( DataReader* stream )
{
   stream->read(_finalized);
}

const String& HashBase::GetName()
{
   return m_handler->name();
}

void HashBase::UpdateData(const String &str)
{
   uint32 size = str.size();
   if( size == 0 )
     return;

   UpdateData( str.getRawStorage(), size );
}

uint64 HashBase::AsInt(void)
{
    byte *digest = GetDigest();
    if(!digest)
        return 0;
    // this is safe, CRC32 and Adler32 (which are 32 bits) have their own implementation.
    // be sure that GetDigest() ALWAYS returns a buffer >= 8 bytes!
    return endianInt64(*((uint64*)digest));
}


//=========================================================================================
// CRC32
//=========================================================================================

uint32 CRC32::_crcTab[256];

CRC32::CRC32(Class* cls):
      HashBase(cls),
      _crc(0xFFFFFFFF)
{
}

CRC32::CRC32(const CRC32& other):
      HashBase(other),
      _crc(other._crc)
{
   if( _finalized ) {
      memcpy( _digest, other._digest, sizeof(_digest) );
   }
}

CRC32::~CRC32()
{}

void CRC32::store(DataWriter* stream) const
{
   HashBase::store(stream);
   stream->write(_crc);
   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}

void CRC32::restore( DataReader* stream )
{
   HashBase::restore(stream);
   stream->read(_crc);
   if( _finalized )
   {
      stream->read( _digest, sizeof(_digest) );
   }
}

void CRC32::GenTab(void)
{
    uint32 crc;
    for (uint16 i = 0; i < 256; i++)
    {
        crc = i;
        for (uint8 j = 8; j > 0; j--)
        {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320L;
            else
                crc >>= 1;
        }
        _crcTab[i] = crc;
    }
}

void CRC32::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    _crc = endianInt32(_crc ^= 0xFFFFFFFF);
    for(uint8 i = 0; i < CRC32_DIGEST_LENGTH; i++)
        _digest[i] = ((byte*)&_crc)[(CRC32_DIGEST_LENGTH-1) - i]; // copy bytes in reverse // TODO: little-endian only?
}

void CRC32::UpdateData( const byte *ptr, uint32 size)
{
    for (uint32 i = 0; i < size; i++)
    {
        _crc = ((_crc >> 8) & 0x00FFFFFF) ^ _crcTab[(_crc ^ *ptr++) & 0xFF];
    }
}

//=========================================================================================
// Adler32
//=========================================================================================


Adler32::Adler32(Class* hdlr) :
         HashBase(hdlr),
         _adler(1)
{
}

Adler32::Adler32(const Adler32& other) :
         HashBase(other),
         _adler(other._adler)
{
   if( _finalized ) {
      memcpy( _digest, other._digest, sizeof(_digest) );
   }
}

Adler32::~Adler32()
{}

void Adler32::store(DataWriter* stream) const
{
   HashBase::store(stream);
   stream->write(_adler);
   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}

void Adler32::restore( DataReader* stream )
{
   HashBase::restore(stream);
   stream->read(_adler);
   if( _finalized )
   {
      stream->read( _digest, sizeof(_digest) );
   }
}

void Adler32::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    uint32 adlerEndian = endianInt32(_adler);
    for(uint8 i = 0; i < ADLER32_DIGEST_LENGTH; i++)
        _digest[i] = ((byte*)&adlerEndian)[(ADLER32_DIGEST_LENGTH-1) - i]; // copy bytes in reverse // TODO: little-endian only?
}

void Adler32::UpdateData( const byte *ptr, uint32 size)
{
    _adler = adler32(_adler, (char*)ptr, size);
}

//=========================================================================================
// SHA1Hash
//=========================================================================================

SHA1Hash::SHA1Hash( Class* hdlr ):
         HashBase(hdlr)
{
    sha_init(&_ctx);
}

SHA1Hash::SHA1Hash( const SHA1Hash& other ):
         HashBase(other)
{
    sha_copy(&_ctx, const_cast<sha_ctx*>(&other._ctx));
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}

SHA1Hash::~SHA1Hash()
{
}


void SHA1Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);
   for( uint32 i = 0; i < SHA_DIGESTLEN; i++ )
   {
      stream->write(_ctx.digest[i]);
   }

   stream->write( _ctx.count_l );
   stream->write( _ctx.count_h );
   stream->writeRaw( _ctx.block, sizeof(_ctx.block) );
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}

void SHA1Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < SHA_DIGESTLEN; i++ )
   {
      stream->read(_ctx.digest[i]);
   }

   stream->read( _ctx.count_l );
   stream->read( _ctx.count_h );
   stream->read( _ctx.block, sizeof(_ctx.block) );
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void SHA1Hash::UpdateData(const byte *ptr, uint32 size)
{
    sha_update(&_ctx, ptr, size);
}


void SHA1Hash::Finalize(void)
{
    if(_finalized)
        return;

    sha_final(&_ctx);
    sha_digest(&_ctx, &_digest[0]);
    _finalized = true;
}

//=========================================================================================
// SHA224Hash
//=========================================================================================


SHA224Hash::SHA224Hash( Class* hdlr ):
         HashBase(hdlr)
{
    sha224_init(&_ctx);
}

SHA224Hash::SHA224Hash( const SHA224Hash& other ):
         HashBase(other)
{
    memcpy(&_ctx, &other._ctx, sizeof(_ctx));
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}

SHA224Hash::~SHA224Hash()
{}


void SHA224Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);
   for( uint32 i = 0; i < _SHA256_SHA224_DIGEST_LENGTH; i++ )
   {
      stream->write(_ctx.state[i]);
   }

   stream->write( _ctx.bitcount );
   stream->writeRaw( _ctx.block, sizeof(_ctx.block) );
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}

void SHA224Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < _SHA256_SHA224_DIGEST_LENGTH; i++ )
   {
      stream->read(_ctx.state[i]);
   }

   stream->read( _ctx.bitcount );
   stream->read( _ctx.block, sizeof(_ctx.block) );
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void SHA224Hash::UpdateData(const byte *ptr, uint32 size)
{
    sha256_sha224_update(&_ctx, ptr, size);
}

void SHA224Hash::Finalize(void)
{
    if(_finalized)
        return;

    sha256_sha224_final(&_ctx);
    sha224_digest(&_ctx, &_digest[0]);
    _finalized = true;
}

//=========================================================================================
// SHA256Hash
//=========================================================================================


SHA256Hash::SHA256Hash( Class* hdlr ):
         HashBase(hdlr)
{
    sha256_init(&_ctx);
}


SHA256Hash::SHA256Hash( const SHA256Hash& other ):
         HashBase(other)
{
    memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}


SHA256Hash::~SHA256Hash()
{
}


void SHA256Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);
   for( uint32 i = 0; i < _SHA256_SHA224_DIGEST_LENGTH; i++ )
   {
      stream->write(_ctx.state[i]);
   }

   stream->write( _ctx.bitcount );
   stream->writeRaw( _ctx.block, sizeof(_ctx.block) );
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}

void SHA256Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < _SHA256_SHA224_DIGEST_LENGTH; i++ )
   {
      stream->read(_ctx.state[i]);
   }

   stream->read( _ctx.bitcount );
   stream->read( _ctx.block, sizeof(_ctx.block) );
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}

void SHA256Hash::UpdateData(const byte *ptr, uint32 size)
{
    sha256_sha224_update(&_ctx, ptr, size);
}

void SHA256Hash::Finalize(void)
{
    if(_finalized)
        return;

    sha256_sha224_final(&_ctx);
    sha256_digest(&_ctx, &_digest[0]);
    _finalized = true;
}


//=========================================================================================
// SHA384Hash
//=========================================================================================


SHA384Hash::SHA384Hash( Class* hdlr ):
         HashBase(hdlr)
{
    sha384_init(&_ctx);
}


SHA384Hash::SHA384Hash( const SHA384Hash& other ):
         HashBase(other)
{
    memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}

SHA384Hash::~SHA384Hash()
{}


void SHA384Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);
   for( uint32 i = 0; i < _SHA512_SHA384_STATE_LENGTH; i++ )
   {
      stream->write(_ctx.state[i]);
   }

   stream->write( _ctx.bitcount_low );
   stream->write( _ctx.bitcount_high );
   stream->writeRaw( _ctx.block, sizeof(_ctx.block) );
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void SHA384Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < _SHA512_SHA384_STATE_LENGTH; i++ )
   {
      stream->read(_ctx.state[i]);
   }

   stream->read( _ctx.bitcount_low );
   stream->read( _ctx.bitcount_high );
   stream->read( _ctx.block, sizeof(_ctx.block) );
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void SHA384Hash::UpdateData(const byte *ptr, uint32 size)
{
    sha512_sha384_update(&_ctx, ptr, size);
}

void SHA384Hash::Finalize(void)
{
    if(_finalized)
        return;

    sha512_sha384_final(&_ctx);
    sha384_digest(&_ctx, &_digest[0]);
    _finalized = true;
}

//=========================================================================================
// SHA384Hash
//=========================================================================================

SHA512Hash::SHA512Hash( Class* hdlr ):
    HashBase(hdlr)
{
    sha512_init(&_ctx);
}

SHA512Hash::SHA512Hash( const SHA512Hash& other ):
         HashBase(other)
{
    memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}

SHA512Hash::~SHA512Hash()
{}


void SHA512Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);
   for( uint32 i = 0; i < _SHA512_SHA384_STATE_LENGTH; i++ )
   {
      stream->write(_ctx.state[i]);
   }

   stream->write( _ctx.bitcount_low );
   stream->write( _ctx.bitcount_high );
   stream->writeRaw( _ctx.block, sizeof(_ctx.block) );
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void SHA512Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < _SHA512_SHA384_STATE_LENGTH; i++ )
   {
      stream->read(_ctx.state[i]);
   }

   stream->read( _ctx.bitcount_low );
   stream->read( _ctx.bitcount_high );
   stream->read( _ctx.block, sizeof(_ctx.block) );
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void SHA512Hash::UpdateData(const byte *ptr, uint32 size)
{
    sha512_sha384_update(&_ctx, ptr, size);
}

void SHA512Hash::Finalize(void)
{
    if(_finalized)
        return;

    sha512_sha384_final(&_ctx);
    sha512_digest(&_ctx, &_digest[0]);
    _finalized = true;
}

//=========================================================================================
// MD2Hash
//=========================================================================================

MD2Hash::MD2Hash(Class* hdlr):
         HashBase(hdlr)
{
    md2_init(&_ctx);
}

MD2Hash::MD2Hash(const MD2Hash& other):
         HashBase(other)
{
    memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
    memcpy(&_digest, &other._digest, sizeof(_digest) );
}

MD2Hash::~MD2Hash()
{}


void MD2Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);

   stream->writeRaw(_ctx.C, sizeof(_ctx.C));
   stream->writeRaw(_ctx.X, sizeof(_ctx.X));
   stream->writeRaw(_ctx.buffer, sizeof(_ctx.buffer));
   stream->write( _ctx.index );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void MD2Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   stream->read(_ctx.C, sizeof(_ctx.C));
   stream->read(_ctx.X, sizeof(_ctx.X));
   stream->read(_ctx.buffer, sizeof(_ctx.buffer));
   stream->read( _ctx.index );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void MD2Hash::UpdateData(const byte *ptr, uint32 size)
{
    md2_update(&_ctx, ptr, size);
}

void MD2Hash::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    md2_digest(&_ctx, _digest);
}

//=========================================================================================
// MD4Hash
//=========================================================================================

MD4Hash::MD4Hash(Class* hdlr):
         HashBase(hdlr)
{
    MD4Init(&_ctx);
}


MD4Hash::MD4Hash(const MD4Hash& other):
                  HashBase(other)
{
   memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
   memcpy(&_digest, &other._digest, sizeof(_digest) );
}


MD4Hash::~MD4Hash()
{}


void MD4Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);

   stream->write(_ctx.buf[0]);
   stream->write(_ctx.buf[1]);
   stream->write(_ctx.buf[2]);
   stream->write(_ctx.buf[3]);

   stream->write(_ctx.bits[0]);
   stream->write(_ctx.bits[1]);

   stream->writeRaw( _ctx.in, sizeof(_ctx.in) );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void MD4Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   stream->read(_ctx.buf[0]);
   stream->read(_ctx.buf[1]);
   stream->read(_ctx.buf[2]);
   stream->read(_ctx.buf[3]);

   stream->read(_ctx.bits[0]);
   stream->read(_ctx.bits[1]);

   stream->read( _ctx.in, sizeof(_ctx.in) );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void MD4Hash::UpdateData(const byte *ptr, uint32 size)
{
    MD4Update(&_ctx, ptr, size);
}

void MD4Hash::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    MD4Final(&_ctx, _digest);
}

//=========================================================================================
// MD5Hash
//=========================================================================================

MD5Hash::MD5Hash(Class* hdlr):
         HashBase(hdlr)
{
    md5_init(&_ctx);
}

MD5Hash::MD5Hash(const MD5Hash& other):
                  HashBase(other)
{
   memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
   memcpy(&_digest, &other._digest, sizeof(_digest) );
}

MD5Hash::~MD5Hash()
{}


void MD5Hash::store(DataWriter* stream) const
{
   HashBase::store(stream);

   stream->write(_ctx.count[0]);
   stream->write(_ctx.count[1]);

   stream->write(_ctx.abcd[0]);
   stream->write(_ctx.abcd[1]);
   stream->write(_ctx.abcd[2]);
   stream->write(_ctx.abcd[3]);

   stream->writeRaw( _ctx.buf, sizeof(_ctx.buf) );

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void MD5Hash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   stream->read(_ctx.count[0]);
   stream->read(_ctx.count[1]);

   stream->read(_ctx.abcd[0]);
   stream->read(_ctx.abcd[1]);
   stream->read(_ctx.abcd[2]);
   stream->read(_ctx.abcd[3]);

   stream->read( _ctx.buf, sizeof(_ctx.buf) );

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void MD5Hash::UpdateData(const byte *ptr, uint32 size)
{
    md5_append(&_ctx, ptr, size);
}

void MD5Hash::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    md5_finish(&_ctx, _digest);
}

//=========================================================================================
// WhirlpoolHash
//=========================================================================================

WhirlpoolHash::WhirlpoolHash( Class* hdlr ):
         HashBase(hdlr)
{
    whirlpool_init(&_ctx);
}


WhirlpoolHash::WhirlpoolHash(const WhirlpoolHash& other):
                  HashBase(other)
{
   memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
   memcpy(&_digest, &other._digest, sizeof(_digest) );
}


WhirlpoolHash::~WhirlpoolHash()
{}


void WhirlpoolHash::store(DataWriter* stream) const
{
   HashBase::store(stream);

   stream->writeRaw(_ctx.bitLength, sizeof(_ctx.bitLength));
   stream->writeRaw(_ctx.buffer, sizeof(_ctx.buffer));
   stream->write(_ctx.bufferBits);
   stream->write(_ctx.bufferPos);
   for( uint32 i = 0; i < DIGESTBYTES/8; ++i )
   {
      stream->write(_ctx.hash[i]);
   }

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void WhirlpoolHash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   stream->read(_ctx.bitLength, sizeof(_ctx.bitLength));
   stream->read(_ctx.buffer, sizeof(_ctx.buffer));
   stream->read(_ctx.bufferBits);
   stream->read(_ctx.bufferPos);
   for( uint32 i = 0; i < DIGESTBYTES/8; ++i )
   {
      stream->read(_ctx.hash[i]);
   }

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void WhirlpoolHash::UpdateData(const byte *ptr, uint32 size)
{
    whirlpool_update(ptr, size * 8, &_ctx); // whirlpool expects size in bits
}

void WhirlpoolHash::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    whirlpool_finalize(&_ctx, _digest);
}

//=========================================================================================
// TigerHash
//=========================================================================================

TigerHash::TigerHash( Class* hdlr ):
    HashBase(hdlr)
{
    tiger_init(&_ctx);
}


TigerHash::TigerHash(const TigerHash& other):
                  HashBase(other)
{
   memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
   memcpy(&_digest, &other._digest, sizeof(_digest) );
}


TigerHash::~TigerHash()
{}


void TigerHash::store(DataWriter* stream) const
{
   HashBase::store(stream);

   stream->write(_ctx.state[0]);
   stream->write(_ctx.state[1]);
   stream->write(_ctx.state[2]);

   stream->write(_ctx.index);

   stream->writeRaw(_ctx.block, sizeof(_ctx.block) );

   stream->write(_ctx.blockcount);

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void TigerHash::restore( DataReader* stream )
{
   HashBase::restore(stream);

   stream->read(_ctx.state[0]);
   stream->read(_ctx.state[1]);
   stream->read(_ctx.state[2]);

   stream->read(_ctx.index);

   stream->read(_ctx.block, sizeof(_ctx.block) );

   stream->read(_ctx.blockcount);

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}


void TigerHash::UpdateData(const byte *ptr, uint32 size)
{
    tiger_update(&_ctx, ptr, size);
}

void TigerHash::Finalize(void)
{
    if(_finalized)
        return;

    _finalized = true;
    tiger_finalize(&_ctx);
    tiger_digest(&_ctx, _digest);
}

//=========================================================================================
// RIPEMDHashBase
//=========================================================================================

RIPEMDHashBase::RIPEMDHashBase(Class* hldr):
   HashBase(hldr)
{}

RIPEMDHashBase::RIPEMDHashBase( const RIPEMDHashBase& other ):
   HashBase(other)
{
   memcpy(&_ctx, &other._ctx, sizeof(_ctx) );
   memcpy(&_digest, &other._digest, sizeof(_digest) );
}

RIPEMDHashBase::~RIPEMDHashBase()
{}


void RIPEMDHashBase::UpdateData(const byte *ptr, uint32 size)
{
    ripemd_update(&_ctx, ptr, size);
}

void RIPEMDHashBase::Finalize(void)
{
    if(_finalized)
        return;

    ripemd_final(&_ctx);
    ripemd_digest(&_ctx, &_digest[0]);
    _finalized = true;
}


void RIPEMDHashBase::store(DataWriter* stream) const
{
   HashBase::store(stream);

   for( uint32 i = 0; i < RIPEMD_STATESIZE; ++i )
   {
      stream->write(_ctx.digest[i]);
   }

   stream->write(_ctx.bitcount);
   stream->writeRaw(_ctx.block, sizeof(_ctx.block));
   stream->write(_ctx.index);
   stream->write(_ctx.digest_len);

   if( _finalized )
   {
      stream->writeRaw( _digest, sizeof(_digest) );
   }
}


void RIPEMDHashBase::restore( DataReader* stream )
{
   HashBase::restore(stream);

   for( uint32 i = 0; i < RIPEMD_STATESIZE; ++i )
   {
      stream->read(_ctx.digest[i]);
   }

   stream->read(_ctx.bitcount);
   stream->read(_ctx.block, sizeof(_ctx.block));
   stream->read(_ctx.index);
   stream->read(_ctx.digest_len);

   if( _finalized )
   {
      stream->read( static_cast<byte*>(_digest), sizeof(_digest) );
   }
}

//=========================================================================================
// RIPEMD128Hash
//=========================================================================================

RIPEMD128Hash::RIPEMD128Hash(Class* hldr):
         RIPEMDHashBase(hldr)
{
    ripemd128_init(&_ctx);
}

RIPEMD128Hash::RIPEMD128Hash( const RIPEMD128Hash& other ):
         RIPEMDHashBase(other)
{}

RIPEMD128Hash::~RIPEMD128Hash()
{}

//=========================================================================================
// RIPEMD160Hash
//=========================================================================================

RIPEMD160Hash::RIPEMD160Hash(Class* hldr):
         RIPEMDHashBase(hldr)
{
    ripemd160_init(&_ctx);
}

RIPEMD160Hash::RIPEMD160Hash(const RIPEMD160Hash& other ):
         RIPEMDHashBase(other)
{
}

RIPEMD160Hash::~RIPEMD160Hash()
{}

//=========================================================================================
// RIPEMD160Hash
//=========================================================================================

RIPEMD256Hash::RIPEMD256Hash(Class* hldr):
         RIPEMDHashBase(hldr)
{
    ripemd256_init(&_ctx);
}

RIPEMD256Hash::RIPEMD256Hash(const RIPEMD256Hash& other ):
         RIPEMDHashBase(other)
{
}

RIPEMD256Hash::~RIPEMD256Hash()
{}

//=========================================================================================
// RIPEMD160Hash
//=========================================================================================

RIPEMD320Hash::RIPEMD320Hash(Class* hldr):
         RIPEMDHashBase(hldr)
{
    ripemd320_init(&_ctx);
}

RIPEMD320Hash::RIPEMD320Hash(const RIPEMD320Hash& other ):
         RIPEMDHashBase(other)
{
}

RIPEMD320Hash::~RIPEMD320Hash()
{}

}
}

/* end of hash_mod.cpp */

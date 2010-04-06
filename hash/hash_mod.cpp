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

#include <falcon/autocstring.h>
#include "hash_mod.h"


namespace Falcon {
namespace Mod {


void HashBase::UpdateData(MemBuf *buf)
{
    uint32 ws = buf->wordSize();
    switch(ws)
    {
        case 1:
            UpdateData(buf->data(), buf->size());
            break;

        case 2:
        case 3:
        case 4:
            for(uint32 i = 0; i < buf->length(); i++)
            {
                int32 c = buf->get(i); // TODO: test this on big endian (conversion done in get(), but not sure if this is correct)
                UpdateData((byte*)&c, ws);
            }
            break;

        default:
            throw new Falcon::TypeError(
                Falcon::ErrorParam( Falcon::e_param_type, __LINE__ )
                .extra( "Unsupported MemBuf word length" ) );
            
    }

}

void HashBase::UpdateData(String *str)
{
    uint32 len = str->length();
    uint32 charSize = str->manipulator()->charSize();
    switch(charSize)
    {
    case 1:
        UpdateData(str->getRawStorage(), len);
        break;

    case 2:
        for(uint32 i = 0; i < len; i++)
        {
            int16 c = endianInt16(str->getCharAt(i)); // TODO: this has to be tested on big-endian!
            UpdateData((byte*)&c, 2);
        }
        break;

    case 4:
        for(uint32 i = 0; i < len; i++)
        {
            int32 c = endianInt32(str->getCharAt(i)); // TODO: this has to be tested on big-endian!
            UpdateData((byte*)&c, 4);
        }
        break;
    }
}

uint32 CRC32::_crcTab[256];

CRC32::CRC32()
: _crc(0xFFFFFFFF)
{
    _finalized = false;
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

void CRC32::UpdateData(byte *ptr, uint32 size)
{
    for (uint32 i = 0; i < size; i++)
    {
        _crc = ((_crc >> 8) & 0x00FFFFFF) ^ _crcTab[(_crc ^ *ptr++) & 0xFF];
    }
}

Adler32::Adler32()
: _adler(1)
{
    _finalized = false;
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

void Adler32::UpdateData(byte *ptr, uint32 size)
{
    _adler = adler32(_adler, (char*)ptr, size);
}

SHA1Hash::SHA1Hash()
{
    _finalized = false;
    SHA1Init(&_ctx);
}

void SHA1Hash::UpdateData(byte *ptr, uint32 size)
{
    SHA1Update(&_ctx, ptr, size);
}

void SHA1Hash::Finalize(void)
{
    if(_finalized)
        return;

    SHA1Final(&_digest[0], &_ctx);
    _finalized = true;
}

SHA224Hash::SHA224Hash()
{
    _finalized = false;
    sha224_init(&_ctx);
}

void SHA224Hash::UpdateData(byte *ptr, uint32 size)
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

SHA256Hash::SHA256Hash()
{
    _finalized = false;
    sha256_init(&_ctx);
}

void SHA256Hash::UpdateData(byte *ptr, uint32 size)
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

SHA384Hash::SHA384Hash()
{
    _finalized = false;
    sha384_init(&_ctx);
}

void SHA384Hash::UpdateData(byte *ptr, uint32 size)
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

SHA512Hash::SHA512Hash()
{
    _finalized = false;
    sha512_init(&_ctx);
}

void SHA512Hash::UpdateData(byte *ptr, uint32 size)
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

MD2Hash::MD2Hash()
{
    _finalized = false;
    md2_init(&_ctx);
}

void MD2Hash::UpdateData(byte *ptr, uint32 size)
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

MD4Hash::MD4Hash()
{
    _finalized = false;
    MD4Init(&_ctx);
}

void MD4Hash::UpdateData(byte *ptr, uint32 size)
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

MD5Hash::MD5Hash()
{
    _finalized = false;
    md5_init(&_ctx);
}

void MD5Hash::UpdateData(byte *ptr, uint32 size)
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

WhirlpoolHash::WhirlpoolHash()
{
    _finalized = false;
    whirlpool_init(&_ctx);
}

void WhirlpoolHash::UpdateData(byte *ptr, uint32 size)
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

TigerHash::TigerHash()
{
    _finalized = false;
    tiger_init(&_ctx);
}

void TigerHash::UpdateData(byte *ptr, uint32 size)
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



}
}


/* end of hash_mod.cpp */

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
#include <falcon/autocstring.h>
#include <string.h>
#include "hash_mod.h"
#include "hash_st.h"


namespace Falcon {
namespace Mod {

HashBase::~HashBase()
{}

// this is a helper function used by makeHash() and the hash() convenience function
FalconData *GetHashByName(String *whichStr)
{
    if(!whichStr->compareIgnoreCase("crc32"))
        return new Mod::HashCarrier<Mod::CRC32>();
    else if(!whichStr->compareIgnoreCase("adler32"))
        return new Mod::HashCarrier<Mod::Adler32>();
    else if(!whichStr->compareIgnoreCase("md2"))
        return new Mod::HashCarrier<Mod::MD2Hash>();
    else if(!whichStr->compareIgnoreCase("md4"))
        return new Mod::HashCarrier<Mod::MD4Hash>();
    else if(!whichStr->compareIgnoreCase("md5"))
        return new Mod::HashCarrier<Mod::MD5Hash>();
    else if(!whichStr->compareIgnoreCase("sha1"))
        return new Mod::HashCarrier<Mod::SHA1Hash>();
    else if(!whichStr->compareIgnoreCase("sha224"))
        return new Mod::HashCarrier<Mod::SHA224Hash>();
    else if(!whichStr->compareIgnoreCase("sha256"))
        return new Mod::HashCarrier<Mod::SHA256Hash>();
    else if(!whichStr->compareIgnoreCase("sha384"))
        return new Mod::HashCarrier<Mod::SHA384Hash>();
    else if(!whichStr->compareIgnoreCase("sha512"))
        return new Mod::HashCarrier<Mod::SHA512Hash>();
    else if(!whichStr->compareIgnoreCase("tiger"))
        return new Mod::HashCarrier<Mod::TigerHash>();
    else if(!whichStr->compareIgnoreCase("whirlpool"))
        return new Mod::HashCarrier<Mod::WhirlpoolHash>();
    else if(!whichStr->compareIgnoreCase("ripemd128"))
        return new Mod::HashCarrier<Mod::RIPEMD128Hash>();
    else if(!whichStr->compareIgnoreCase("ripemd160"))
        return new Mod::HashCarrier<Mod::RIPEMD160Hash>();
    else if(!whichStr->compareIgnoreCase("ripemd256"))
        return new Mod::HashCarrier<Mod::RIPEMD256Hash>();
    else if(!whichStr->compareIgnoreCase("ripemd320"))
        return new Mod::HashCarrier<Mod::RIPEMD320Hash>();

    // note: when adding entries here, be sure to overload the hash's GetName() method accordingly!

    return NULL;
}

CoreString *ByteArrayToHex(byte *arr, uint32 size)
{
    CoreString *str = new CoreString; // each byte will be encoded to 2 chars
    str->reserve(size * 2);

    for(uint32 i = 0; i < size; i++)
    {
        int hexlet = (arr[i] >> 4) & 0xf ;
        str->append( hexlet < 10 ? '0' + hexlet : 'a' + (hexlet-10) );
        hexlet = arr[i] & 0xf ;
        str->append( hexlet < 10 ? '0' + hexlet : 'a' + (hexlet-10) );
    }
    return str;
}


void HashBase::UpdateData(MemBuf *buf)
{
    uint32 ws = buf->wordSize();
    switch(ws)
    {
        case 1:
            UpdateData(buf->data() + buf->position(), buf->limit() - buf->position());
            break;

        case 2:
        case 3:
        case 4:
            for(uint32 i = buf->position(); i < buf->limit(); i++)
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


HashBaseFalcon::HashBaseFalcon()
: _bytes(0), _digest(NULL), _intval(0)
{
    _finalized = false;
}

HashBaseFalcon::~HashBaseFalcon()
{
    if(_digest)
        delete [] _digest;
}

void HashBaseFalcon::_GetCallableMethod(Falcon::Item& item, const Falcon::String& name)
{
    if(!_self->getMethod(name, item))
    {
        throw new Falcon::AccessError(
            Falcon::ErrorParam( Falcon::e_miss_iface, __LINE__ )
            .extra( name ) );
    }
    if(!item.isCallable())
    {
        throw new Falcon::AccessError(
        Falcon::ErrorParam( Falcon::e_non_callable, __LINE__ )
        .extra( name ) );
    }
}

void HashBaseFalcon::Finalize(void)
{
    if(_finalized)
        return;

    Falcon::Item m;
    _GetCallableMethod(m, "finalize");
    _vm->callItemAtomic(m, 0);
    _finalized = true; // assume success only if it didn't throw
}

uint32 HashBaseFalcon::DigestSize(void)
{
    if(!_bytes) // cache the byte count so the call has to be made only once
    {
        Falcon::Item m;
        _GetCallableMethod(m, "bytes"); // this is safe since bytes() is overloaded
        _vm->callItemAtomic(m, 0);
        _bytes = Falcon::uint32(_vm->regA().forceIntegerEx()); // throws if returned not a number
        if(!_bytes)
        {
            throw new Falcon::GenericError(
                Falcon::ErrorParam( Falcon::e_prop_invalid, __LINE__ )
                .extra(_vm->moduleString(hash_err_size)));
        }
    }
    return _bytes;
}

byte *HashBaseFalcon::GetDigest(void)
{
    // if we have already cached our digest, return that
    if(_digest)
        return _digest;

    // otherwise, calculate it
    if(!IsFinalized())
        Finalize();

    Falcon::Item m;
    _GetCallableMethod(m, "toMemBuf"); // this is safe since toMemBuf() is overloaded
    _vm->callItemAtomic(m, 0);
    Falcon::Item ret = _vm->regA(); // copy item, the check against DigestSize() might overwrite the reference otherwise
    if( !(ret.isMemBuf() && ret.asMemBuf() && ret.asMemBuf()->wordSize() == 1) )
    {
        throw new Falcon::GenericError(
            Falcon::ErrorParam( Falcon::e_prop_invalid, __LINE__ )
            .extra(_vm->moduleString(hash_err_not_membuf_1)));
    }

    // this check is maybe not necessary, but enforces a more correct implementation of overloaded hash classes
    uint32 s = DigestSize();
    if( ret.asMemBuf()->length() != s )
    {
        throw new Falcon::GenericError(
            Falcon::ErrorParam( Falcon::e_prop_invalid, __LINE__ )
            .extra(_vm->moduleString(hash_err_membuf_length_differs)));
    }

    // copy the result, in case the GC eats up the MemBuf
    _digest = new byte[s];
    memcpy(_digest, ret.asMemBuf()->data(), s);
    return _digest;
}

void HashBaseFalcon::UpdateData( const byte *ptr, uint32 size)
{
    if(!size)
        return;
    Falcon::Item m;
    _GetCallableMethod(m, "process");
    Falcon::MemBuf_1 *mb = new Falcon::MemBuf_1( (byte*)ptr, size, 0);
    _vm->pushParam(mb);
    _vm->callItemAtomic(m, 1);
}

uint64 HashBaseFalcon::AsInt(void)
{
    // cached?
    if(_intval)
        return _intval;

    // HashBase::AsInt() expects the buffer to be at least 8 bytes long
    uint32 s = DigestSize();
    if(s >= sizeof(uint64))
        return HashBase::AsInt();

    // buffer is smaller, process manually
    uint64 val = 0;
    byte *valp = (byte*)&val;
    byte *digest = GetDigest();
    for(uint32 i = 0; i < s; ++i)
        valp[i] = digest[i];
    _intval = endianInt64(val);
    return _intval;
}




uint32 CRC32::_crcTab[256];

CRC32::CRC32()
: _crc(0xFFFFFFFF)
{
    _finalized = false;
}

CRC32::~CRC32()
{}

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

Adler32::Adler32()
: _adler(1)
{
    _finalized = false;
}

Adler32::~Adler32()
{}

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

SHA1Hash::SHA1Hash()
{
    _finalized = false;
    sha_init(&_ctx);
}

SHA1Hash::~SHA1Hash()
{}


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

SHA224Hash::SHA224Hash()
{
    _finalized = false;
    sha224_init(&_ctx);
}


SHA224Hash::~SHA224Hash()
{}

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

SHA256Hash::SHA256Hash()
{
    _finalized = false;
    sha256_init(&_ctx);
}

SHA256Hash::~SHA256Hash()
{}


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

SHA384Hash::SHA384Hash()
{
    _finalized = false;
    sha384_init(&_ctx);
}

SHA384Hash::~SHA384Hash()
{}

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

SHA512Hash::SHA512Hash()
{
    _finalized = false;
    sha512_init(&_ctx);
}

SHA512Hash::~SHA512Hash()
{}

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

MD2Hash::MD2Hash()
{
    _finalized = false;
    md2_init(&_ctx);
}

MD2Hash::~MD2Hash()
{}


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

MD4Hash::MD4Hash()
{
    _finalized = false;
    MD4Init(&_ctx);
}

MD4Hash::~MD4Hash()
{}

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

MD5Hash::MD5Hash()
{
    _finalized = false;
    md5_init(&_ctx);
}

MD5Hash::~MD5Hash()
{}

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

WhirlpoolHash::WhirlpoolHash()
{
    _finalized = false;
    whirlpool_init(&_ctx);
}


WhirlpoolHash::~WhirlpoolHash()
{}


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

TigerHash::TigerHash()
{
    _finalized = false;
    tiger_init(&_ctx);
}

TigerHash::~TigerHash()
{}

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

RIPEMD128Hash::RIPEMD128Hash()
{
    _finalized = false;
    ripemd128_init(&_ctx);
}

RIPEMD128Hash::~RIPEMD128Hash()
{}

RIPEMD160Hash::RIPEMD160Hash()
{
    _finalized = false;
    ripemd160_init(&_ctx);
}

RIPEMD160Hash::~RIPEMD160Hash()
{}

RIPEMD256Hash::RIPEMD256Hash()
{
    _finalized = false;
    ripemd256_init(&_ctx);
}

RIPEMD256Hash::~RIPEMD256Hash()
{}

RIPEMD320Hash::RIPEMD320Hash()
{
    _finalized = false;
    ripemd320_init(&_ctx);
}

RIPEMD320Hash::~RIPEMD320Hash()
{}


}
}


/* end of hash_mod.cpp */

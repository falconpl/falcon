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
   Internal logic functions - declarations.
*/

#ifndef FALCON_FEATHERS_HASH_MOD_H
#define FALCON_FEATHERS_HASH_MOD_H

#include <falcon/types.h>
#include "adler32.h"
#include "sha1.h"
#include "sha256_sha224.h"
#include "sha512_sha384.h"
#include "md2.h"
#include "md4.h"
#include "md5.h"
#include "whirlpool.h"
#include "tiger.h"
#include "ripemd.h"

#define CRC32_DIGEST_LENGTH       4  // 32 bits
#define ADLER32_DIGEST_LENGTH     4  // 32 bits
#define SHA1_DIGEST_LENGTH       20  // 160 bits
#define SHA224_DIGEST_LENGTH     28  // 224 bits
#define SHA256_DIGEST_LENGTH     32  // 256 bits
#define SHA384_DIGEST_LENGTH     48  // 384 bits
#define SHA512_DIGEST_LENGTH     64  // 512 bits
#define MD2_DIGEST_LENGTH        16  // 128 bits
#define MD4_DIGEST_LENGTH        16  // 128 bits
#define MD5_DIGEST_LENGTH        16  // 128 bits
#define WHIRLPOOL_DIGEST_LENGTH  64  // 512 bits
#define TIGER_DIGEST_LENGTH      24  // 192 bits
#define RIPEMD128_DIGEST_LENGTH  16  // 128 bits
#define RIPEMD160_DIGEST_LENGTH  20  // 160 bits
#define RIPEMD256_DIGEST_LENGTH  32  // 256 bits
#define RIPEMD320_DIGEST_LENGTH  40  // 320 bits

// should there be any hash that has a greater block size don't forget to change this!!
#define MAX_USED_BLOCKSIZE 128

namespace Falcon {

class DataWriter;
class DataReader;
class Class;

namespace Mod {

    class HashBase
    {
    public:
        virtual ~HashBase( );
        // these are the same for every hash function and should not be overloaded
        void UpdateData(const String& str);
        inline bool IsFinalized(void) { return _finalized; }

        // each hashing algorithm must overload DigestSize(), UpdateData(), Finalize(), and GetDigest()
        virtual void UpdateData(const byte *ptr, uint32 size) {}
        virtual void Finalize(void) {}
        virtual uint32 DigestSize(void) { return 0; }

        virtual byte *GetDigest(void) { return NULL; }

        // required for HMAC, since 64 is the block size for most hashes we take that as default and override only where necessary
        virtual uint32 GetBlockSize(void) { return 64; }

        // can be overloaded optionally (but MUST be overloaded if DigestSize < 8)
        virtual uint64 AsInt(void);

        // Resolves to taking the name of the handler class.
        const String& GetName(void);

        void gcMark( uint32 m ) {m_gcMark = m; }
        uint32 currentMark() const { return m_gcMark; }

        Class* handler() const { return m_handler; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );

    protected:
        bool _finalized;
        HashBase( Class* hldr );
        HashBase( const HashBase& other );

    private:
        uint32 m_gcMark;
        Class* m_handler;
    };


    class CRC32 : public HashBase
    {
    public:
        CRC32(Class* hldr);
        CRC32(const CRC32 &other);
        virtual ~CRC32();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return CRC32_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint64 AsInt(void) { return _finalized ? _crc : 0; } // special for CRC32
        static void GenTab(void);

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        uint32 _crc;
        byte _digest[CRC32_DIGEST_LENGTH];
        static uint32 _crcTab[256];
    };

    class Adler32 : public HashBase
    {
    public:
        Adler32(Class* hldr);
        Adler32(const Adler32& other);
        virtual ~Adler32();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return ADLER32_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint64 AsInt(void) { return _finalized ? _adler : 0; } // special for Adler32

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        uint32 _adler;
        byte _digest[ADLER32_DIGEST_LENGTH];
    };

    class SHA1Hash : public HashBase
    {
    public:
        SHA1Hash(Class* hldr);
        SHA1Hash(const SHA1Hash& other);
        virtual ~SHA1Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA1_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        sha_ctx _ctx;
        byte _digest[SHA1_DIGEST_LENGTH];
    };

    class SHA224Hash : public HashBase
    {
    public:
        SHA224Hash(Class* hldr);
        SHA224Hash(const SHA224Hash& other);
        virtual ~SHA224Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA224_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        sha256_sha224_ctx _ctx;
        byte _digest[SHA224_DIGEST_LENGTH];
    };

    class SHA256Hash : public HashBase
    {
    public:
        SHA256Hash(Class* hldr);
        SHA256Hash(const SHA256Hash &other);
        virtual ~SHA256Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA256_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        sha256_sha224_ctx _ctx;
        byte _digest[SHA256_DIGEST_LENGTH];
    };

    class SHA384Hash : public HashBase
    {
    public:
        SHA384Hash(Class* hldr);
        SHA384Hash( const SHA384Hash& other );
        virtual ~SHA384Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA384_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint32 GetBlockSize(void) { return 128; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        sha512_sha384_ctx _ctx;
        byte _digest[SHA384_DIGEST_LENGTH];
    };

    class SHA512Hash : public HashBase
    {
    public:
        SHA512Hash(Class* hldr);
        SHA512Hash( const SHA512Hash& other );
        virtual ~SHA512Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA512_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint32 GetBlockSize(void) { return 128; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        sha512_sha384_ctx _ctx;
        byte _digest[SHA512_DIGEST_LENGTH];
    };

    class MD2Hash : public HashBase
    {
    public:
        MD2Hash(Class* hldr);
        MD2Hash( const MD2Hash& other );
        virtual ~MD2Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD2_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        md2_ctx _ctx;
        byte _digest[MD2_DATA_SIZE];
    };

    class MD4Hash : public HashBase
    {
    public:
        MD4Hash(Class* hldr);
        MD4Hash(const MD4Hash& other);
        virtual ~MD4Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD4_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        MD4_CTX _ctx;
        byte _digest[MD4_DIGEST_LENGTH];
    };

    class MD5Hash : public HashBase
    {
    public:
        MD5Hash(Class* hldr);
        MD5Hash( const MD5Hash& other );
        virtual ~MD5Hash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD5_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        md5_state_t _ctx;
        byte _digest[MD5_DIGEST_LENGTH];
    };

    class WhirlpoolHash : public HashBase
    {
    public:
        WhirlpoolHash(Class* hldr);
        WhirlpoolHash(const WhirlpoolHash& other);
        virtual ~WhirlpoolHash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return WHIRLPOOL_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        whirlpool_ctx _ctx;
        byte _digest[WHIRLPOOL_DIGEST_LENGTH];
    };

    class TigerHash : public HashBase
    {
    public:
        TigerHash(Class* hldr);
        TigerHash(const TigerHash& other);
        virtual ~TigerHash();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return TIGER_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );
    private:
        tiger_ctx _ctx;
        byte _digest[TIGER_DIGEST_LENGTH];
    };

    class RIPEMDHashBase : public HashBase
    {
    public:

        virtual ~RIPEMDHashBase();
        void UpdateData(const byte *ptr, uint32 size);
        void Finalize(void);
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

        virtual void store( DataWriter* stream) const;
        virtual void restore( DataReader* stream );

    protected:
        RIPEMDHashBase(Class* hldr);
        RIPEMDHashBase( const RIPEMDHashBase& other );

        ripemd_ctx _ctx;
        byte _digest[RIPEMD320_DIGEST_LENGTH];
    };

    class RIPEMD128Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD128Hash(Class* hldr);
        RIPEMD128Hash( const RIPEMD128Hash& other );
        virtual ~RIPEMD128Hash();
        uint32 DigestSize(void) { return RIPEMD128_DIGEST_LENGTH; }

    };

    class RIPEMD160Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD160Hash(Class* hldr);
        RIPEMD160Hash(const RIPEMD160Hash& other);
        virtual ~RIPEMD160Hash();
        uint32 DigestSize(void) { return RIPEMD160_DIGEST_LENGTH; }
    };

    class RIPEMD256Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD256Hash(Class* hldr);
        RIPEMD256Hash( const RIPEMD256Hash& other );
        virtual ~RIPEMD256Hash();
        uint32 DigestSize(void) { return RIPEMD256_DIGEST_LENGTH; }
    };

    class RIPEMD320Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD320Hash(Class* hldr);
        RIPEMD320Hash( const RIPEMD320Hash& other );
        virtual ~RIPEMD320Hash();
        uint32 DigestSize(void) { return RIPEMD320_DIGEST_LENGTH; }
    };

}
}

#endif

/* end of hash_mod.h */

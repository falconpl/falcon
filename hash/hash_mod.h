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

#ifndef hash_mod_H
#define hash_mod_H

#include <falcon/types.h>
#include <falcon/membuf.h>
#include <falcon/falcondata.h>
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

namespace Falcon {
namespace Mod {

    class HashBase
    {
    public:
        // these are the same for every hash function and should not be overloaded
        void UpdateData(MemBuf *buf);
        void UpdateData(String *str);
        inline bool IsFinalized(void) { return _finalized; }

        // each hashing algorithm must overload DigestSize(), UpdateData(), Finalize(), and GetDigest()
        virtual void UpdateData(byte *ptr, uint32 size) {}
        virtual void Finalize(void) {}
        virtual uint32 DigestSize(void) { return 0; }
        virtual byte *GetDigest(void) { return NULL; }

        // can be overloaded optionally (but MUST be overloaded if DigestSize < 8)
        virtual uint64 AsInt(void);

    protected:
        bool _finalized;
    };

    class HashBaseFalcon : public HashBase
    {
    public:
        HashBaseFalcon();
        ~HashBaseFalcon();
        virtual void UpdateData(byte *ptr, uint32 size); // VM call to process() in overloaded falcon class
        virtual void Finalize(void); // VM call to internal_finalize() in overloaded falcon class
        virtual uint32 DigestSize(void); // VM call to bytes() in overloaded falcon class
        virtual byte *GetDigest(void); // VM call to toMemBuf() in overloaded falcon class
        virtual uint64 AsInt(void);

        inline void SetVM(Falcon::VMachine *vm) { _vm = vm; }

    protected:
        void _GetCallableMethod(Falcon::Item& item, const Falcon::String& name);
        VMachine *_vm;
        uint32 _bytes; // caches bytes() value
        byte *_digest; // stores a copy of toMemBuf() value once called
        uint64 _intval; // caches toInt() value
    };

    class CRC32 : public HashBase
    {
    public:
        CRC32();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return CRC32_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint64 AsInt(void) { return _finalized ? _crc : 0; } // special for CRC32
        static void GenTab(void);

    private:
        uint32 _crc;
        byte _digest[CRC32_DIGEST_LENGTH];
        static uint32 _crcTab[256];
    };

    class Adler32 : public HashBase
    {
    public:
        Adler32();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return ADLER32_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }
        uint64 AsInt(void) { return _finalized ? _adler : 0; } // special for Adler32

    private:
        uint32 _adler;
        byte _digest[ADLER32_DIGEST_LENGTH];
    };

    class SHA1Hash : public HashBase
    {
    public:
        SHA1Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA1_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        SHA1_CTX _ctx;
        byte _digest[SHA1_DIGEST_LENGTH];
    };

    class SHA224Hash : public HashBase
    {
    public:
        SHA224Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA224_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        sha256_sha224_ctx _ctx;
        byte _digest[SHA224_DIGEST_LENGTH];
    };

    class SHA256Hash : public HashBase
    {
    public:
        SHA256Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA256_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        sha256_sha224_ctx _ctx;
        byte _digest[SHA256_DIGEST_LENGTH];
    };

    class SHA384Hash : public HashBase
    {
    public:
        SHA384Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA384_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        sha512_sha384_ctx _ctx;
        byte _digest[SHA384_DIGEST_LENGTH];
    };

    class SHA512Hash : public HashBase
    {
    public:
        SHA512Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return SHA512_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        sha512_sha384_ctx _ctx;
        byte _digest[SHA512_DIGEST_LENGTH];
    };

    class MD2Hash : public HashBase
    {
    public:
        MD2Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD2_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        md2_ctx _ctx;
        byte _digest[MD2_DATA_SIZE];
    };

    class MD4Hash : public HashBase
    {
    public:
        MD4Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD4_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        MD4_CTX _ctx;
        byte _digest[MD4_DIGEST_LENGTH];
    };

    class MD5Hash : public HashBase
    {
    public:
        MD5Hash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return MD5_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        md5_state_t _ctx;
        byte _digest[MD5_DIGEST_LENGTH];
    };

    class WhirlpoolHash : public HashBase
    {
    public:
        WhirlpoolHash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return WHIRLPOOL_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        whirlpool_ctx _ctx;
        byte _digest[WHIRLPOOL_DIGEST_LENGTH];
    };

    class TigerHash : public HashBase
    {
    public:
        TigerHash();
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        uint32 DigestSize(void) { return TIGER_DIGEST_LENGTH; }
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    private:
        tiger_ctx _ctx;
        byte _digest[TIGER_DIGEST_LENGTH];
    };

    class RIPEMDHashBase : public HashBase
    {
    public:
        void UpdateData(byte *ptr, uint32 size);
        void Finalize(void);
        byte *GetDigest(void) { return _finalized ? &_digest[0] : NULL; }

    protected:
        ripemd_ctx _ctx;
        byte _digest[RIPEMD320_DIGEST_LENGTH];
    };

    class RIPEMD128Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD128Hash();
        uint32 DigestSize(void) { return RIPEMD128_DIGEST_LENGTH; }
    };

    class RIPEMD160Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD160Hash();
        uint32 DigestSize(void) { return RIPEMD160_DIGEST_LENGTH; }
    };

    class RIPEMD256Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD256Hash();
        uint32 DigestSize(void) { return RIPEMD256_DIGEST_LENGTH; }
    };

    class RIPEMD320Hash : public RIPEMDHashBase
    {
    public:
        RIPEMD320Hash();
        uint32 DigestSize(void) { return RIPEMD320_DIGEST_LENGTH; }
    };



    template <class HASH> class HashCarrier : public FalconData
    {
    public:
        HashCarrier() {};
        inline HASH *GetHash(void) { return &hash; }

        virtual HashCarrier<HASH> *clone() const { return NULL; } // not cloneable

        virtual void gcMark( uint32 mark ) {}

        virtual bool serialize( Stream *stream, bool bLive ) const { return false; }
        virtual bool deserialize( Stream *stream, bool bLive ) { return false; }
    private:
        HASH hash;
    };


}
}

#endif

/* end of hash_mod.h */

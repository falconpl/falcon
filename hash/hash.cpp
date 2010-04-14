/*
   FALCON - The Falcon Programming Language.
   FILE: hash_ext.cpp

   Provides multiple hashing algorithms
   Main module file, providing the module object to
   the Falcon engine.
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
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include <falcon/symbol.h>
#include "hash_mod.h"
#include "hash_ext.h"
#include "hash_srv.h"
#include "hash_st.h"

#include "version.h"

/*#
    @module feather_hash hash
    @brief Various hash and checksum functions

    This module provides a selection of the most widely used checksum/hash algorithms:
    CRC32, Adler32, SHA-1, SHA-224, SHA-256, SHA-384, SHA-512, MD2, MD4, MD5, Whirlpool, Tiger

    @beginmodule feathers_hash
*/

/*#
    @group checksums Checksums
    @brief Classes providing checksum functions

    This group of classes provides simple checksum functions to verify integrity of arbitrary data.
    They are NOT meant for use in cryptographic algorithms or @b safe data verification!
*/

/*#
    @group weak_hashes Weak hashes
    @brief Classes providing weak / deprecated hashes

    This group of classes provides hashes that are stronger (and longer) then checksums,
    but not recommended for serious cryptographic purposes (MD2, MD4, MD5 and partly SHA1 can be considered broken).
*/

/*#
    @group strong_hashes Strong hashes
    @brief Classes providing strong hashes, suitable for cryptography

    Hashes in this group are cryptographically strong and can be used for @b secure verification of data.
*/

/*#
    @class CRC32
    @ingroup checksums
    @brief Calculates a 32 bits long CRC32 checksum
*/

/*#
    @class Adler32
    @ingroup checksums
    @brief Calculates a 32 bits long Adler32 checksum
*/

/*#
    @class SHA1Hash
    @ingroup weak_hashes
    @brief Calculates a 160 bits long SHA-1 hash
*/

/*#
    @class MD2Hash
    @ingroup weak_hashes
    @brief Calculates a 128 bits long MD2 (Message Digest 2) hash
*/

/*#
    @class MD4Hash
    @ingroup weak_hashes
    @brief Calculates a 128 bits long MD4 (Message Digest 4) hash
*/

/*#
    @class MD5Hash
    @ingroup weak_hashes
    @brief Calculates a 128 bits long MD5 (Message Digest 5) hash
*/

/*#
    @class SHA224Hash
    @ingroup strong_hashes
    @brief Calculates a 224 bits long SHA224 hash (SHA-2 family)
*/

/*#
    @class SHA256Hash
    @ingroup strong_hashes
    @brief Calculates a 256 bits long SHA256 hash (SHA-2 family)
*/

/*#
    @class SHA384Hash
    @ingroup strong_hashes
    @brief Calculates a 384 bits long SHA384 hash (SHA-2 family)
*/

/*#
    @class SHA512Hash
    @ingroup strong_hashes
    @brief Calculates a 512 bits long SHA512 hash (SHA-2 family)
*/

/*#
    @class TigerHash
    @ingroup strong_hashes
    @brief Calculates a 192 bits long Tiger hash
*/

/*#
    @class WhirlpoolHash
    @ingroup strong_hashes
    @brief Calculates a 512 bits long Whirlpool hash
*/

template <class HASH> Falcon::Symbol *SimpleRegisterHash(Falcon::Module *self, const char *name)
{
    Falcon::Symbol *cls = self->addClass(name, Falcon::Ext::Hash_init<HASH>);
    self->addClassMethod(cls, "update", Falcon::Ext::Hash_update<HASH>);
    self->addClassMethod(cls, "updateInt",   Falcon::Ext::Hash_updateInt<HASH>).asSymbol()->
        addParam("num")->addParam("bytes");
    self->addClassMethod(cls, "finalize",   Falcon::Ext::Hash_finalize<HASH>);
    self->addClassMethod(cls, "isFinalized",Falcon::Ext::Hash_isFinalized<HASH>);
    self->addClassMethod(cls, "bytes", Falcon::Ext::Hash_bytes<HASH>);
    self->addClassMethod(cls, "bits", Falcon::Ext::Hash_bits<HASH>);
    self->addClassMethod(cls, "toMemBuf",  Falcon::Ext::Hash_toMemBuf<HASH>);
    self->addClassMethod(cls, "toString",   Falcon::Ext::Hash_toString<HASH>);

    return cls;
}

Falcon::Module *hash_module_init(void)
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "hash" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // Here declare the international string table implementation
   //
   #include "hash_st.h"

   //============================================================
   // API declarations
   //
   Falcon::Symbol *crc32_cls = SimpleRegisterHash<Falcon::Mod::CRC32>(self, "CRC32");
   self->addClassMethod(crc32_cls, "toInt", Falcon::Ext::Hash_toInt<Falcon::Mod::CRC32>);

   Falcon::Symbol *adler32_cls = SimpleRegisterHash<Falcon::Mod::Adler32>(self, "Adler32");
   self->addClassMethod(adler32_cls, "toInt", Falcon::Ext::Hash_toInt<Falcon::Mod::Adler32>);

   SimpleRegisterHash<Falcon::Mod::HashBaseFalcon>(self, "HashBase"     );
   SimpleRegisterHash<Falcon::Mod::SHA1Hash>      (self, "SHA1Hash"     );
   SimpleRegisterHash<Falcon::Mod::SHA224Hash>    (self, "SHA224Hash"   );
   SimpleRegisterHash<Falcon::Mod::SHA256Hash>    (self, "SHA256Hash"   );
   SimpleRegisterHash<Falcon::Mod::SHA384Hash>    (self, "SHA384Hash"   );
   SimpleRegisterHash<Falcon::Mod::SHA512Hash>    (self, "SHA512Hash"   );
   SimpleRegisterHash<Falcon::Mod::MD2Hash>       (self, "MD2Hash"      );
   SimpleRegisterHash<Falcon::Mod::MD4Hash>       (self, "MD4Hash"      );
   SimpleRegisterHash<Falcon::Mod::MD5Hash>       (self, "MD5Hash"      );
   SimpleRegisterHash<Falcon::Mod::WhirlpoolHash> (self, "WhirlpoolHash");
   SimpleRegisterHash<Falcon::Mod::TigerHash>     (self, "TigerHash"    );

   // generate CRC32 table
   Falcon::Mod::CRC32::GenTab();

   return self;
}

FALCON_MODULE_DECL
{
    return hash_module_init();
}

/* end of hash.cpp */

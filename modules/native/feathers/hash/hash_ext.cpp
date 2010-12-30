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
Interface extension functions
*/

/*#
@beginmodule feathers.hash
*/

#include <stdio.h>
#include <string.h>
#include <falcon/engine.h>
#include <falcon/autocstring.h>

#include "hash_mod.h"
#include "hash_st.h"
#include "hash_ext.h"


namespace Falcon {
namespace Ext {



/*#
@function getSupportedHashes
@brief Returns an array containing the names of all supported hashes.
@return An array containing the names of all hashes supported by this module

This function can be used to check if different versions of the module support a specific hash.
*/
FALCON_FUNC Func_GetSupportedHashes( ::Falcon::VMachine *vm )
{
    CoreArray *arr = new CoreArray(16);
    arr->append(new CoreString("CRC32"));
    arr->append(new CoreString("Adler32"));
    arr->append(new CoreString("SHA1"));
    arr->append(new CoreString("SHA224"));
    arr->append(new CoreString("SHA256"));
    arr->append(new CoreString("SHA384"));
    arr->append(new CoreString("SHA512"));
    arr->append(new CoreString("MD2"));
    arr->append(new CoreString("MD4"));
    arr->append(new CoreString("MD5"));
    arr->append(new CoreString("Tiger"));
    arr->append(new CoreString("Whirlpool"));
    arr->append(new CoreString("RIPEMD128"));
    arr->append(new CoreString("RIPEMD160"));
    arr->append(new CoreString("RIPEMD256"));
    arr->append(new CoreString("RIPEMD320"));
    vm->retval(arr);
}

/*#
@function hash
@brief Convenience function that calculates a hash.
@param raw If set to true, return a raw MemBuf instead of a string.
@param which Hash that should be used
@optparam data... Arbitrary amount of parameters that should be fed into the chosen hash function.
@return A lowercase hexadecimal string with the output of the chosen hash if @i raw is false, or a 1-byte wide MemBuf if true.
@raise ParamError in case @i which is a string and a hash with that name was not found; or if @i which is not a hash object.
@raise AccessError if @i which is a hash object that was already finalized.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

Param @i which can contain a String with the name of the hash, a hash class constructor, a not-finalized hash object,
or a function that returns any of the latter.

@note Use getSupportedHashes() to check which hash names are supported.
@note If @i which is a hash object, it will be finalized by calling this function.
*/
FALCON_FUNC Func_hash( ::Falcon::VMachine *vm )
{
    if(vm->paramCount() < 2)
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( "B, X, [, X...]" ) );
    }
    
    bool raw = vm->param(0)->asBoolean();
    Item which = *(vm->param(1));
    
    Mod::HashCarrier<Mod::HashBase> *carrier = NULL;

    while(which.isCallable())
    {
        vm->callItemAtomic(which, 0);
        which = vm->regA();
    }

    bool ownCarrier = false;

    if(which.isString())
    {
        carrier = (Mod::HashCarrier<Mod::HashBase>*)(Mod::GetHashByName(which.asString()));
        ownCarrier = true;
    }
    else if(which.isObject())
    {
        CoreObject *co = which.asObject();
        if(co->derivedFrom("HashBase"))
            carrier = (Mod::HashCarrier<Mod::HashBase>*)(co->getUserData());
    }

    if(!carrier)
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( FAL_STR(hash_not_found) ) );
    }

    Mod::HashBase *hash = carrier->GetHash();

    for(uint32 i = 2; i < uint32(vm->paramCount()); i++)
    {
        Item *what = vm->param(i);
        if (!what)
        {
            throw new Falcon::ParamError( 
                Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
                .extra( "A|S|M" ) );
        }
        Hash_updateItem_internal(what, hash, vm, 0);
    }

    hash->Finalize();

    uint32 size = hash->DigestSize();
    byte *digest = hash->GetDigest();

    if(raw)
    {
        Falcon::MemBuf_1 *buf = new Falcon::MemBuf_1(size);
        memcpy(buf->data(), digest, size);
        vm->retval(buf);
    }
    else
    {
        vm->retval(Mod::ByteArrayToHex(digest, size));
    }

    if(ownCarrier)
        delete carrier;
}

/*#
@function makeHash
@brief Creates a hash object based on the algorithm name
@param name The name of the algorithm (case insensitive)
@optparam silent Return @b nil instead of throwing an exception, if the hash was not found
@return A new instance of a hash object of the the given algorithm name 
@raise ParamError in case a hash with that name was not found
@note Use getSupportedHashes() to check which hash names are supported.
*/
FALCON_FUNC Func_makeHash( ::Falcon::VMachine *vm )
{
    if(vm->paramCount() < 1 || !vm->param(0)->isString())
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( "S" ) );
    }
    
    String *name = vm->param(0)->asString();
    bool silent =  vm->paramCount() > 1 && vm->param(1)->asBoolean();
    FalconData *fdata = Mod::GetHashByName(name);
    Mod::HashCarrier<Mod::HashBase> *carrier = (Mod::HashCarrier<Mod::HashBase>*)(fdata);
    if(!carrier)
    {
        if(silent)
        {
            vm->retnil();
            return;
        }
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).desc( FAL_STR(hash_not_found) ).extra(*name) );
    }

    Item *wki = vm->findWKI(carrier->GetHash()->GetName());
    if(!wki)
    {   // this should NOT happen.. most possibly a hash name was not set correctly
        throw new GenericError( ErrorParam( e_noninst_cls, __LINE__ )
            .origin( e_orig_mod ).extra( FAL_STR(hash_internal_error) ) );
    }
    CoreClass *cls = wki->asClass();

    FalconObject *obj = new FalconObject(cls);
    obj->setUserData(carrier);
    vm->retval(obj);
}

/*#
@function hmac
@brief Provides HMAC authentication for a block of data
@param raw If set to true, return a raw MemBuf instead of a string.
@param which Hash that should be used
@param key Secret authentication key
@param data The data to be authenticated
@return A lowercase hexadecimal string with the HMAC-result of the chosen hash if @i raw is false, or a 1-byte wide MemBuf if true.
@raise ParamError in case @i which is a string and a hash with that name was not found; or if @i which does not evaluate to a hash object.
@raise AccessError if @i which evaluates to a hash object that was already finalized.

Param @i key can be a String or a MemBuf.

Param @i which can contain a String with the name of the hash, a hash class constructor, or a function that returns a useable hash object.
Unlike the hash() function, it is not possible to pass a hash object directly, because it would have to be used 3 times, which is not possible
because of finalization.

In total, this function evaluates @i which 3 times, creating 3 hash objects internally.

@note Use getSupportedHashes() to check which hash names are supported.
*/
FALCON_FUNC Func_hmac( ::Falcon::VMachine *vm )
{
    Item *i_raw = vm->param(0);
    Item *i_which = vm->param(1);
    Item *i_key = vm->param(2);
    Item *i_data = vm->param(3);
    if( !(i_raw && i_which && i_key)
     || !(i_which->isCallable() || i_which->isString())
     || !(i_key->isMemBuf() || i_key->isString())
     || !(i_data->isMemBuf() || i_data->isString()) )
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( "B, X, X, [, X...]" ) );
    }

    bool raw = i_raw->asBoolean();

    // first, get the 3 hash items to use
    Item hashItem[3];
    bool ownCarrier[3] = { false, false, false };
    Mod::HashCarrier<Mod::HashBase> *carrier[3] = { NULL, NULL, NULL };
    Mod::HashBase *hash[3] = { NULL, NULL, NULL };
    bool success = true;
    for(uint32 i = 0; i < 3; ++i)
    {
        Item& itemRef = hashItem[i];
        itemRef = *i_which;
        while(itemRef.isCallable())
        {
            vm->callItemAtomic(itemRef, 0);
            itemRef = vm->regA();
        }
        if(itemRef.isString())
        {
            carrier[i] = (Mod::HashCarrier<Mod::HashBase>*)(Mod::GetHashByName(itemRef.asString()));
            ownCarrier[i] = true;
        }
        else if(itemRef.isObject())
        {
            CoreObject *co = itemRef.asObject();
            if(co->derivedFrom("HashBase"))
                carrier[i] = (Mod::HashCarrier<Mod::HashBase>*)(co->getUserData());
        }
        if(carrier[i])
            hash[i] = carrier[i]->GetHash();
        else
            success = false;
    }

    if(success)
    {
        uint32 blocksize = hash[0]->GetBlockSize();
        uint32 byteCount;

        byte i_key_pad[MAX_USED_BLOCKSIZE];
        byte o_key_pad[MAX_USED_BLOCKSIZE];

        if(i_key->isMemBuf())
        {
            MemBuf *buf = i_key->asMemBuf();
            byteCount = buf->size();
        }
        else
        {
            String *str = i_key->asString();
            byteCount = str->size();
        }
        
        if(byteCount > blocksize) // key too large? hash it, so the resulting size will be equal to blocksize
        {
            if(i_key->isString())
                hash[0]->UpdateData(*i_key->asString());
            else
                hash[0]->UpdateData(i_key->asMemBuf());
            hash[0]->Finalize();
            byte *digest = hash[0]->GetDigest();
            uint32 digestSize = hash[0]->DigestSize();
            memcpy(i_key_pad, digest, digestSize);
            memcpy(o_key_pad, digest, digestSize);
            if ( digestSize < blocksize )
            {
               memset(i_key_pad + digestSize, 0, blocksize - digestSize);
               memset(o_key_pad + digestSize, 0, blocksize - digestSize);
            }
            
        }
        else if(byteCount <= blocksize) // key too small? if the key has exactly blocksize bytes we can go this way too
        {
            // TODO: is the way the memory is accessed here ok? works on big endian? different char sizes in strings?
            uint32 remain = blocksize - byteCount;
            byte *memptr;
            if(i_key->isMemBuf()) // it's a MemBuf, copy into output buffers and pad with zeros
            {
                MemBuf *buf = i_key->asMemBuf();
                memptr = buf->data();
            }
            else // it's a string, append zeros
            {
                String *str = i_key->asString();
                memptr = str->getRawStorage();
            }
            memcpy(i_key_pad, memptr, blocksize);
            memcpy(o_key_pad, memptr, blocksize);
            if(remain)
            {
                memset(i_key_pad + byteCount, 0, remain);
                memset(o_key_pad + byteCount, 0, remain);
            }
        }

        for(uint32 i = 0; i < blocksize; ++i)
        {
            o_key_pad[i] ^= 0x5C;
            i_key_pad[i] ^= 0x36;
        }

        // inner hash
        hash[1]->UpdateData(i_key_pad, blocksize);
        if(i_data->isString())
            hash[1]->UpdateData(*i_data->asString());
        else
            hash[1]->UpdateData(i_data->asMemBuf());
        hash[1]->Finalize();

        // outer hash
        hash[2]->UpdateData(o_key_pad, blocksize);
        hash[2]->UpdateData(hash[1]->GetDigest(), hash[1]->DigestSize());
        hash[2]->Finalize();

        uint32 size = hash[2]->DigestSize();
        byte *digest = hash[2]->GetDigest();

        if(raw)
        {
            Falcon::MemBuf_1 *buf = new Falcon::MemBuf_1(size);
            memcpy(buf->data(), digest, size);
            vm->retval(buf);
        }
        else
        {
            vm->retval(Mod::ByteArrayToHex(digest, size));
        }
    }

    // cleanup
    for(uint32 i = 0; i < 3; ++i)
        if(ownCarrier[i])
            delete carrier[i];

    if(!success)
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( FAL_STR(hash_not_found) ) );
    }
}



// updateItem is a helper function to process the individual items passed to update()
void Hash_updateItem_internal(Item *what, Mod::HashBase *hash, ::Falcon::VMachine *vm, uint32 stackDepth)
{
    if(stackDepth > 500) // TODO: is this value safe? does it require adjusting for other platforms/OSes?
    {
        throw new Falcon::GenericError(
            Falcon::ErrorParam( Falcon::e_stackof, __LINE__ )
            .extra( "Too deep recursion, aborting" ) );
    }

    Item method;
    if(what->isMemBuf())
    {
        hash->UpdateData(what->asMemBuf());
    }
    else if(what->isString())
    {
        hash->UpdateData(*what->asString());
    }
    else if(what->isArray())
    {
        CoreArray *arr = what->asArray();
        for(uint32 i = 0; i < arr->length(); ++i)
        {
            Hash_updateItem_internal(&arr->at(i), hash, vm, stackDepth + 1);
        }
    }
    else if(what->isDict())
    {
        CoreDict *dict = what->asDict();
        Iterator iter(&dict->items());
        while( iter.hasCurrent() )
        {
            Hash_updateItem_internal(&iter.getCurrent(), hash, vm, stackDepth + 1);
            iter.next();
        }
    }
    else if(what->isOfClass("List"))
    {
        ItemList *li = dyncast<ItemList *>( what->asObject()->getSequence() );
        Iterator iter(li);
        while( iter.hasCurrent() )
        {
            Hash_updateItem_internal(&iter.getCurrent(), hash, vm, stackDepth + 1);
            iter.next();
        }
    }
    // skip nil, hashing it as string "Nil" would be useless and error-prone
    else if(what->isNil())
    {
        return;
    }
    else if(what->isObject() && what->asObject()->getMethod("toMemBuf", method) && method.isCallable())
    {
        vm->callItemAtomic(method, 0);
        Item mb = vm->regA();
        // whatever we got as result, hash it. it does not necessarily have to be a MemBuf for this to work.
        Hash_updateItem_internal(&mb, hash, vm, stackDepth + 1);
    }
    /*else if(what->isOfClass("MPZ")) // direct conversion from MPZ to hash -- as soon as MPZ provide toMemBuf, this can be dropped
    {
        Item mpz;
        // involve the VM to convert an MPZ to string in base 16
        // and then convert that into the individual bytes beeing hashed (backwards, to represent the original number)
        // i have checked it and the way this is done here is *correct*!
        if(what->asObject()->getMethod("toString", mpz))
        {
            vm->pushParameter(16);
            vm->callItemAtomic(mpz, 1);
            String *hexstr = vm->regA().asString();
            if(uint32 len = hexstr->length())
            {
                char tmp[3];
                tmp[2] = 0;
                uint32 maxlen = (len & 1) ? len - 1 : len; // skip leftmost byte if string length is uneven
                byte b;

                for(uint32 i = 0 ; i < maxlen ; i += 2)
                {
                    tmp[0] = hexstr->getCharAt((len - 1) - (i + 1));
                    tmp[1] = hexstr->getCharAt((len - 1) - i);
                    b = (byte)strtoul(tmp, NULL, 16); // converting max. 0xFF, this is safe
                    hash->UpdateData(&b, 1);
                }
                if(len & 1) // something remaining? must be treated as if it was prepended by '0'
                {
                    tmp[0] = hexstr->getCharAt(0);
                    tmp[1] = 0;
                    b = (byte)strtoul(tmp, NULL, 16);
                    hash->UpdateData(&b, 1);
                }
            }
        }
        else
        {
            throw new Falcon::AccessError(
                Falcon::ErrorParam( Falcon::e_miss_iface, __LINE__ )
                .extra( "MPZ does not provide toString, blame OmniMancer" ) );
        }
    }*/
    else // fallback - convert to string if nothing else works
    {
        String str;
        what->toString( str );
        hash->UpdateData( str );
    }
}


}
}


// some massive documentation below
// it is put here not to bloat the other files too much

/*#
@function adler32
@ingroup checksums
@brief Convenience function that calculates a 32 bits long Adler32 checksum
@return A lowercase hexadecimal string with the adler32 checksum.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function adler32(...)
        hash = Adler32()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a Adler32 object, nor finalization.

@see Adler32.update
*/

/*#
@function sha1
@ingroup weak_hashes
@brief Convenience function that calculates a 160 bits long SHA1 hash
@return A lowercase hexadecimal string with the SHA1 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function sha1(...)
        hash = SHA1Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a SHA1Hash object, nor finalization.

@see SHA1Hash.update
*/

/*#
@function sha224
@ingroup strong_hashes
@brief Convenience function that calculates a 224 bits long SHA224 hash
@return A lowercase hexadecimal string with the SHA224 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function sha224(...)
        hash = SHA224Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a SHA224Hash object, nor finalization.

@see SHA224Hash.update
*/

/*#
@function sha256
@ingroup strong_hashes
@brief Convenience function that calculates a 256 bits long SHA256 hash
@return A lowercase hexadecimal string with the SHA256 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function sha256(...)
        hash = SHA256Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a SHA256Hash object, nor finalization.

@see SHA256Hash.update
*/

/*#
@function sha384
@ingroup strong_hashes
@brief Convenience function that calculates a 384 bits long SHA384 hash
@return A lowercase hexadecimal string with the SHA384 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function sha384(...)
        hash = SHA384Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a SHA384Hash object, nor finalization.

@see SHA384Hash.update
*/

/*#
@function sha512
@ingroup strong_hashes
@brief Convenience function that calculates a 512 bits long SHA512 hash
@return A lowercase hexadecimal string with the SHA512 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function sha512(...)
        hash = SHA512Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a SHA512Hash object, nor finalization.

@see SHA512Hash.update
*/

/*#
@function md2
@ingroup weak_hashes
@brief Convenience function that calculates a 128 bits long MD2 hash
@return A lowercase hexadecimal string with the MD2 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function md2(...)
        hash = MD2Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a MD2Hash object, nor finalization.

@see MD2Hash.update
*/

/*#
@function md4
@ingroup weak_hashes
@brief Convenience function that calculates a 128 bits long MD4 hash
@return A lowercase hexadecimal string with the MD4 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function md4(...)
        hash = MD4Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a MD4Hash object, nor finalization.

@see MD4Hash.update
*/

/*#
@function md5
@ingroup weak_hashes
@brief Convenience function that calculates a 128 bits long MD5 hash
@return A lowercase hexadecimal string with the MD5 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function md5(...)
        hash = MD5Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a MD5Hash object, nor finalization.

@see MD5Hash.update
*/

/*#
@function tiger
@ingroup strong_hashes
@brief Convenience function that calculates a 192 bits long Tiger hash
@return A lowercase hexadecimal string with the Tiger hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function tiger(...)
        hash = TigerHash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a TigerHash object, nor finalization.

@see TigerHash.update
*/

/*#
@function whirlpool
@ingroup strong_hashes
@brief Convenience function that calculates a 512 bits long Whirlpool hash
@return A lowercase hexadecimal string with the Whirlpool hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function whirlpool(...)
        hash = WhirlpoolHash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a WhirlpoolHash object, nor finalization.

@see WhirlpoolHash.update
*/

/*#
@function ripemd128
@ingroup weak_hashes
@brief Convenience function that calculates a 128 bits long RIPEMD128 hash
@return A lowercase hexadecimal string with the RIPEMD128 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function ripemd128(...)
        hash = RIPEMD128Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a RIPEMD128Hash object, nor finalization.

@see RIPEMD128Hash.update
*/

/*#
@function ripemd160
@ingroup strong_hashes
@brief Convenience function that calculates a 160 bits long RIPEMD160 hash
@return A lowercase hexadecimal string with the RIPEMD160 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function ripemd160(...)
        hash = RIPEMD160Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a RIPEMD160Hash object, nor finalization.

@see RIPEMD160Hash.update
*/

/*#
@function ripemd256
@ingroup strong_hashes
@brief Convenience function that calculates a 256 bits long RIPEMD256 hash.
@return A lowercase hexadecimal string with the RIPEMD256 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function ripemd256(...)
        hash = RIPEMD256Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a RIPEMD256Hash object, nor finalization.

@see RIPEMD256Hash.update
*/

/*#
@function ripemd320
@ingroup strong_hashes
@brief Convenience function that calculates a 320 bits long RIPEMD320 hash
@return A lowercase hexadecimal string with the RIPEMD320 hash.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function ripemd320(...)
        hash = RIPEMD320Hash()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a RIPEMD320Hash object, nor finalization.

@see RIPEMD320Hash.update
*/

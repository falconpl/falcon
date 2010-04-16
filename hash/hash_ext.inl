/*
   FALCON - The Falcon Programming Language.
   FILE: hash_ext.inl

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
@beginmodule feather_hash
*/

#include <stdio.h>
#include <string.h>
#include <falcon/engine.h>
#include "hash_mod.h"
#include "hash_ext.h"
#include "hash_st.h"

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


template <class HASH> FALCON_FUNC Hash_init( ::Falcon::VMachine *vm )
{
    vm->self().asObject()->setUserData(new Mod::HashCarrier<HASH>);
}

// specialization required to correctly assign the VM
template <> FALCON_FUNC Hash_init<Mod::HashBaseFalcon>( ::Falcon::VMachine *vm )
{
    Mod::HashCarrier<Mod::HashBaseFalcon> *carrier = new Mod::HashCarrier<Mod::HashBaseFalcon>;
    carrier->GetHash()->SetVM(vm);
    vm->self().asObject()->setUserData(carrier);
}

/*#
@method update HashBase
@brief Feeds data into the hash function.
@raise AccessError if the hash is already finalized.
@raise GenericError in case of a stack overflow. This can happen with self- or circular references inside objects.
@return The object itself.

This method accepts an @i arbitrary amount of parameters, each treated differently:
- Strings and MemBufs are hashed with respect to their byte count (1, 2, or 4 byte strings) and endianness.
- MPZ (Big Numbers) are converted to a hex string internally, which is then converted to a byte array and hashed (this operation is endian-neutral).
- Lists and Arrays are traversed, each item beeing hashed.
- Dictionaries: only the values are hashed (the keys not). Note that the order in which the values are processed depends on the keys!
- Nil as parameter is always skipped, even if contained in a Sequence.
- If a parameter provides a @b toMemBuf method, it is called and the returned result hashed (does not have to be a MemBuf, actually).
  This allows direct hashing of finalized hashes.
- In all other cases, the parameter is converted to a string.
- To hash integer values, use @b updateInt(), as it respects their memory layout. If put into @b update(), they will be hashed as a @i string!
- All parameters are hashed in the order they are passed.
- Sequences can be nested.

@note @b update() can be called arbitrarily often, until @b finalize() has been called.
@note Multiple calls can be chained, e.g. hash.update(x).update(y).update(z)
*/

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
        hash->UpdateData(what->asString());
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
    else if(what->isOfClass("MPZ")) // direct conversion from MPZ to hash -- as soon as MPZ provide toMemBuf, this can be dropped
    {
        Item *mpz = new Item;
        // involve the VM to convert an MPZ to string in base 16
        // and then convert that into the individual bytes beeing hashed (backwards, to represent the original number)
        // i have checked it and the way this is done here is *correct*!
        if(what->asObject()->getMethod("toString", *mpz))
        {
            vm->pushParameter(16);
            vm->callItemAtomic(*mpz, 1);
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
            delete mpz;
            throw new Falcon::AccessError(
                Falcon::ErrorParam( Falcon::e_miss_iface, __LINE__ )
                .extra( "MPZ does not provide toString, blame OmniMancer" ) );
        }
        delete mpz;
    }
    else // fallback - convert to string if nothing else works
    {
        String *str = new String();
        what->toString(*str);
        hash->UpdateData(str);
        delete str;
    }
}

template <class HASH> FALCON_FUNC Hash_update( ::Falcon::VMachine *vm )
{
    Mod::HashCarrier<HASH> *carrier = (Mod::HashCarrier<HASH>*)(vm->self().asObject()->getUserData());
    Mod::HashBase *hash = carrier->GetHash();
    if(hash->IsFinalized())
    {
        throw new Falcon::AccessError( 
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra(FAL_STR(hash_err_finalized)));
    }
    for(uint32 i = 0; i < uint32(vm->paramCount()); i++)
    {
        Item *what = vm->param(i);
        if (!what)
        {
            throw new Falcon::ParamError( 
                Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
                .extra( "MemBuf or S or Array" ) );
        }
        Hash_updateItem_internal(what, hash, vm, 0);
    }

    vm->retval(vm->self());
}

/*#
@method updateInt HashBase
@brief Hashes an integer of a specified byte length.
@param num The integer value to hash.
@param bytes The amount of bytes to take.
@raise ParamError if @b num is not a number or @b bytes is not in 1..8
@raise AccessError if the hash is already finalized.
@return The object itself.

This method can be used to avoid creating a MemBuf to hash integer values.
It supports 1 up to 8 bytes (uint64).

All integers are internally converted to little-endian. Floating-point numbers are automatically converted to Integers, all other types raise an error.

@note @b updateInt() can be called arbitrarily often, until @b finalize() has been called.
@note Multiple calls can be chained, e.g. hash.updateInt(x).updateInt(y).updateInt(z)
*/
template <class HASH> FALCON_FUNC Hash_updateInt( ::Falcon::VMachine *vm )
{
    Mod::HashCarrier<HASH> *carrier = (Mod::HashCarrier<HASH>*)(vm->self().asObject()->getUserData());
    Mod::HashBase *hash = carrier->GetHash();
    if(hash->IsFinalized())
    {
        throw new Falcon::AccessError( 
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra(FAL_STR(hash_err_finalized)));
    }
    if(vm->paramCount() < 2)
    {
        throw new Falcon::ParamError( 
            Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
            .extra( "N, N" ) );
    }
    uint64 num = vm->param(0)->forceIntegerEx();
    uint8 bytes = (uint8)vm->param(1)->forceIntegerEx();
    if( !(bytes && bytes <= 8) )
    {
        throw new Falcon::ParamError( 
            Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
            .extra( "bytes must be in 1..8" ) );
    }
    num = endianInt64(num);
    hash->UpdateData((byte*)&num, bytes);

    vm->retval(vm->self());
}

/*#
@method finalize HashBase
@brief Finalizes a hash and produces the actual result.
@return The object itself.

A hash object instance must be finalized before the result can be obtained.
After finalizing, no more data can be added.

@note Does nothing if the hash is already finalized.
@note It is possible to add another call after @b finalize(), e.g. hash.finalize().toString()
*/
template <class HASH> FALCON_FUNC Hash_finalize( ::Falcon::VMachine *vm )
{
    Mod::HashCarrier<HASH> *carrier = (Mod::HashCarrier<HASH>*)(vm->self().asObject()->getUserData());
    Mod::HashBase *hash = carrier->GetHash();
    if(hash->IsFinalized())
        return;

    hash->Finalize();
    vm->retval(vm->self());
}

/*#
@method isFinalized HashBase
@brief Checks if a hash is finalized.
@return true if the hash is finalized, false if not.
*/
template <class HASH> FALCON_FUNC Hash_isFinalized( ::Falcon::VMachine *vm )
{
    vm->retval(((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash()->IsFinalized());
}

/*#
@method bytes HashBase
@brief Returns the byte length of the hash result.
@return The amount of @b bytes of the hash result.

The amount of returned bytes is specific for each hash algorithm.
Does not require finalize() called previously.
*/
template <class HASH> FALCON_FUNC Hash_bytes( ::Falcon::VMachine *vm )
{
    vm->retval((Falcon::int32)((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash()->DigestSize());
}

// specialization to return 0 for not overloaded HashBaseFalcon
// it is required because HashBaseFalcon::DigestSize() invokes a VM call to the bytes() method,
// which would call DigestSize() again...
template <> FALCON_FUNC Hash_bytes<Mod::HashBaseFalcon>( ::Falcon::VMachine *vm )
{
    vm->retval(Falcon::int32(0));
}

/*#
@method bits HashBase
@brief Returns the bit length of the hash result.
@return The amount of @b bits of the hash result.

The bit length of a hash function is a rough indicator for its safety - long hashes take exponentially longer to find a collision,
or to break them.

@note This method is a shortcut for @b bytes() * 8
*/
template <class HASH> FALCON_FUNC Hash_bits( ::Falcon::VMachine *vm )
{
    vm->retval((Falcon::int32)((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash()->DigestSize() * 8);
}

/*#
@method toMemBuf HashBase
@brief Returns the hash result in a MemBuf.
@return The hash result, in a 1-byte wide MemBuf.
@raise AccessError if the hash is not finalized.

Before calling this, the hash must be finalized.
*/
template <class HASH> FALCON_FUNC Hash_toMemBuf( ::Falcon::VMachine *vm )
{
    Mod::HashBase *hash = ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash();
    uint32 size = hash->DigestSize();
    Falcon::MemBuf_1 *buf = new Falcon::MemBuf_1(size);
    if(byte *digest = hash->GetDigest())
    {
        memcpy(buf->data(), digest, size);
        vm->retval(buf);
    }
    else
    {
        throw new Falcon::AccessError( 
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra(FAL_STR(hash_err_not_finalized)));
    }
}

template <> FALCON_FUNC Hash_toMemBuf<Mod::HashBaseFalcon>( ::Falcon::VMachine *vm )
{
    throw new Falcon::GenericError( 
        Falcon::ErrorParam( Falcon::e_miss_iface, __LINE__ )
        .extra(vm->moduleString(hash_err_no_overload)));
}

/*#
@method toString HashBase
@brief Returns the hash result as a hexadecimal string.
@return The hash result, as a lowercased hexadecimal string.

Before calling this, the hash must be finalized. 
*/
template <class HASH> FALCON_FUNC Hash_toString( ::Falcon::VMachine *vm )
{
    Mod::HashBase *hash = ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash();
    uint32 size = hash->DigestSize();
    if(byte *digest = hash->GetDigest())
    {
        Falcon::String *str = new Falcon::String(size * 2); // each byte will be encoded to 2 chars
        char tmp[3];

        for(uint32 i = 0; i < size; i++)
        {
            sprintf(tmp, "%02x", digest[i]); // convert byte to hex
            str->A(tmp[0]).A(tmp[1]); // and add it to output string
        }

        vm->retval(str);
        return;
    }
    else
    {
        throw new Falcon::AccessError( 
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra(FAL_STR(hash_err_not_finalized)));
    }

}

/*#
@method toInt HashBase
@brief Returns the result as an Integer value.
@return The checksum result, as an Integer.

Converts up to 8 bytes from the actual hash result to an integer value and returns it, depending on its length.
If the hash is longer, the 8 lowest bytes are taken. (MemBuf[0] to MemBuf[7])

Before calling this, the hash must be finalized. 
@note The returned int is in native endianness.
*/
template <class HASH> FALCON_FUNC Hash_toInt( ::Falcon::VMachine *vm )
{
    vm->retval((Falcon::int64)((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash()->AsInt());
}


}
}

/* end of hash_mod.inl */

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

#include <string.h>
#include <falcon/engine.h>
#include "hash_mod.h"
#include "hash_ext.h"
#include "hash_st.h"

namespace Falcon {
namespace Ext {


template <class HASH> FALCON_FUNC Hash_init( ::Falcon::VMachine *vm )
{
    vm->self().asObject()->setUserData(new Mod::HashCarrier<HASH>);
}

// specialization required to correctly assign the VM
template <> FALCON_FUNC Hash_init<Mod::HashBaseFalcon>( ::Falcon::VMachine *vm )
{
    Mod::HashCarrier<Mod::HashBaseFalcon> *carrier = new Mod::HashCarrier<Mod::HashBaseFalcon>();
    Mod::HashBaseFalcon *hash = carrier->GetHash();
    hash->SetVM(vm);
    hash->SetSelf(vm->self().asObject());
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
- Lists and Arrays are traversed, each item beeing hashed.
- Dictionaries: only the values are hashed (the keys not). Note that the order in which the values are processed depends on the keys!
- Nil as parameter is always skipped, even if contained in a Sequence.
- If a parameter provides a @b toMemBuf method, it is called and the returned result hashed (does not have to be a MemBuf, actually).
  This allows direct hashing of finalized hashes.
- In all other cases, the parameter is converted to a string.
- To hash integer values, use @b updateInt(), as it respects their memory layout. If put into @b update(), they will be hashed as a @i string!
- All parameters are hashed in the order they are passed.
- Sequences can be nested.

@note Multiple calls can be chained, e.g. hash.update(x).update(y).update(z)
*/
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
@method isFinalized HashBase
@brief Checks if a hash is finalized.
@return true if the hash is finalized, false if not.

When a result from a hash is obtained, the hash will be finalized, making it impossible to add additional data.
This method can be used if the finalization state of a hash is unknown.
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

@note Calling this method will finalize the hash.
*/
template <class HASH> FALCON_FUNC Hash_toMemBuf( ::Falcon::VMachine *vm )
{
    Mod::HashBase *hash = ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash();
    if(!hash->IsFinalized())
        hash->Finalize();
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
            .extra(FAL_STR(hash_err_no_digest)));
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

@note Calling this method will finalize the hash.
*/
template <class HASH> FALCON_FUNC Hash_toString( ::Falcon::VMachine *vm )
{
    Mod::HashBase *hash = ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash();
    if(!hash->IsFinalized())
        hash->Finalize();
    uint32 size = hash->DigestSize();
    if(byte *digest = hash->GetDigest())
    {
        vm->retval(Mod::ByteArrayToHex(digest, size));
    }
    else
    {
        throw new Falcon::AccessError( 
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra(FAL_STR(hash_err_no_digest)));
    }
}

/*#
@method toInt HashBase
@brief Returns the result as an Integer value.
@return The checksum result, as an Integer.

Converts up to 8 bytes from the actual hash result to an integer value and returns it, depending on its length.
If the hash is longer, the 8 lowest bytes are taken. (MemBuf[0] to MemBuf[7])

@note Calling this method will finalize the hash. 
@note The returned int is in native endianness.
*/
template <class HASH> FALCON_FUNC Hash_toInt( ::Falcon::VMachine *vm )
{
    Mod::HashBase *hash = ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->GetHash();
    if(!hash->IsFinalized())
        hash->Finalize();
    vm->retval((Falcon::int64)hash->AsInt());
}

/*#
@function crc32
@ingroup checksums
@brief Convenience function that calculates a 32 bits long CRC32 checksum
@return A lowercase hexadecimal string with the crc32 checksum.

This function takes an arbitrary amount of parameters. See HashBase.update() for details.

The semantics are equal to:
@code
    function crc32(...)
        hash = CRC32()
        hash.update(...)
        hash.finalize()
        return hash.toString()
    end
@endcode

@note This is a shortcut function that does neither require creation of a CRC32 object, nor finalization.

@see CRC32.update
*/

// documentation for similar hash shortcut functions follows at the end of hash_ext.cpp

template <class HASH> FALCON_FUNC Func_hashSimple( ::Falcon::VMachine *vm )
{
    HASH hash;

    for(uint32 i = 0; i < uint32(vm->paramCount()); i++)
    {
        Item *what = vm->param(i);
        if (!what)
        {
            throw new Falcon::ParamError( 
                Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
                .extra( "MemBuf or S or Array" ) );
        }
        Hash_updateItem_internal(what, &hash, vm, 0);
    }

    hash.Finalize();

    vm->retval(Mod::ByteArrayToHex(hash.GetDigest(), hash.DigestSize()));
}

/*#
@method reset HashBase
@brief Clears the hash state

Clears a hash and sets it back to the state when it was created.
*/
template <class HASH> FALCON_FUNC Hash_reset( ::Falcon::VMachine *vm )
{
    ((Mod::HashCarrier<HASH>*)vm->self().asObject()->getUserData())->Reset();
}


}
}


/* end of hash_ext.inl */

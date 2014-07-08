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

#define SRC "modules/native/feathers/hash_ext.cpp"

#include <stdio.h>
#include <string.h>
#include <falcon/engine.h>
#include <falcon/autocstring.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/stderrors.h>
#include <falcon/extfunc.h>
#include <falcon/itemarray.h>
#include <falcon/vmcontext.h>
#include <falcon/common.h>

#include "hash_mod.h"
#include "hash_fm.h"


namespace Falcon {
namespace Feathers {


class ClassHash: public Class
{
public:
   ClassHash();
   virtual ~ClassHash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
   virtual void dispose( void* instance ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;

   bool op_init( VMContext*, void*, int32 ) const;

protected:
   ClassHash(const String& name, Class* parent);
};

// updateItem is a helper function to process the individual items passed to update()
static void Hash_updateItem_internal(Item *what, Mod::HashBase *hash, ::Falcon::VMContext *vm, uint32 stackDepth)
{
    if(stackDepth > 256) // TODO: is this value safe? does it require adjusting for other platforms/OSes?
    {
        throw new Falcon::GenericError(
            Falcon::ErrorParam( Falcon::e_stackof, __LINE__ )
            .extra( "Too deep recursion, aborting" ) );
    }

    Item method;

    if(what->isString())
    {
        hash->UpdateData(*what->asString());
    }
    else if( what->isOrdinal() )
    {
       byte b = static_cast<byte>(what->forceInteger());
       hash->UpdateData(&b,1);
    }
    else if(what->isArray())
    {
        ItemArray *arr = what->asArray();
        for(uint32 i = 0; i < arr->length(); ++i)
        {
            Hash_updateItem_internal(&arr->at(i), hash, vm, stackDepth + 1);
        }
    }
    // skip nil, hashing it as string "Nil" would be useless and error-prone
    else if(what->isNil())
    {
        return;
    }

    else // fallback - convert to string if nothing else works
    {
       throw FALCON_SIGN_XERROR( Falcon::ParamError, e_param_type,
                   .extra( "Invalid type input in hashing function/method" ) );
    }
}


/*#
@function getSupportedHashes
@brief Returns an array containing the names of all supported hashes.
@return An array containing the names of all hashes supported by this module

This function can be used to check if different versions of the module support a specific hash.
*/
FALCON_FUNC Func_GetSupportedHashes( ::Falcon::VMContext *ctx, int32 )
{
    ItemArray *arr = new ItemArray(16);
    arr->append(FALCON_GC_HANDLE(new String("CRC32")));
    arr->append(FALCON_GC_HANDLE(new String("Adler32")));
    arr->append(FALCON_GC_HANDLE(new String("SHA1")));
    arr->append(FALCON_GC_HANDLE(new String("SHA224")));
    arr->append(FALCON_GC_HANDLE(new String("SHA256")));
    arr->append(FALCON_GC_HANDLE(new String("SHA384")));
    arr->append(FALCON_GC_HANDLE(new String("SHA512")));
    arr->append(FALCON_GC_HANDLE(new String("MD2")));
    arr->append(FALCON_GC_HANDLE(new String("MD4")));
    arr->append(FALCON_GC_HANDLE(new String("MD5")));
    arr->append(FALCON_GC_HANDLE(new String("Tiger")));
    arr->append(FALCON_GC_HANDLE(new String("Whirlpool")));
    arr->append(FALCON_GC_HANDLE(new String("RIPEMD128")));
    arr->append(FALCON_GC_HANDLE(new String("RIPEMD160")));
    arr->append(FALCON_GC_HANDLE(new String("RIPEMD256")));
    arr->append(FALCON_GC_HANDLE(new String("RIPEMD320")));
    ctx->retval(FALCON_GC_HANDLE(arr));
}


static void internal_hash( Function* caller, VMContext* ctx, int32 pcount, bool isRaw )
{
   ModuleHash* mod = static_cast<ModuleHash*>(caller->module());
   if( pcount < 2 )
   {
       throw caller->paramError(__LINE__, SRC );
   }
    
    Item which = *(ctx->param(1));
    
    Class* cls = 0;

    if(which.isString())
    {
        cls = mod->getClass(*which.asString());
    }
    else if(which.isClass())
    {
       cls = static_cast<Class*>(which.asInst());
       if(! cls->isDerivedFrom( mod->m_baseHashCls ) )
       {
          throw caller->paramError( "hash parameter is not a Hash class", __LINE__, SRC );
       }
    }
    else {
       throw caller->paramError(__LINE__, SRC );
    }

    Mod::HashBase *hash = static_cast<Mod::HashBase*>(cls->createInstance());

    try
    {
       for( int32 i = 1; i < pcount; i++)
       {
           Item *what = ctx->param(i);
           Hash_updateItem_internal(what, hash, ctx, 0);
       }

       hash->Finalize();

       uint32 size = hash->DigestSize();
       byte *digest = hash->GetDigest();

       String* str = new String;
       if( isRaw )
       {
          str->adoptMemBuf(digest,size,0);
          str->toMemBuf(); // internally copy
       }
       else {
          Mod::hashToString(*str, false, digest, size);
          ctx->returnFrame();
       }

       ctx->returnFrame(FALCON_GC_HANDLE(str));
       cls->dispose(hash);
    }
    catch( ... )
    {
       cls->dispose(hash);
       throw;
    }
}
/*#
@function hash
@brief Convenience function that calculates a hash.
@param which Hash class (or hash name) that should be used.
@optparam data... Arbitrary amount of parameters that should be fed into the chosen hash function.
@return A lowercase hexadecimal string with the output of the chosen hash.
@raise ParamError in case @b which is a string and a hash with that name was not found; or if @v which is not a hash class.

The @b which parameter is either a hash class or the name of a hashing class provided by
this module. A new instance of the hashing class is internally created, used and then disposed.

This function takes an arbitrary amount of parameters. Each parameter past @b which is fed
into the @a Hash.update method of the selected hash class.

@note Use getSupportedHashes() to check which hash names are supported.

@note To get the class of a hash instance use @a BOM.baseClass.
*/
FALCON_DECLARE_FUNCTION(hash, "hash:S|Class,data:S,...")
FALCON_DEFINE_FUNCTION_P(hash)
{
   internal_hash( this, ctx, pCount, false );
}

/*#
@function hash_r
@brief Convenience function that calculates a hash.
@param which Hash class (or hash name) that should be used.
@optparam data... Arbitrary amount of parameters that should be fed into the chosen hash function.
@return A memory buffer string with the output of the chosen hash.
@raise ParamError in case @b which is a string and a hash with that name was not found; or if @v which is not a hash class.

The @b which parameter is either a hash class or the name of a hashing class provided by
this module. A new instance of the hashing class is internally created, used and then disposed.

This function takes an arbitrary amount of parameters. Each parameter past @b which is fed
into the @a Hash.update method of the selected hash class.

@note Use getSupportedHashes() to check which hash names are supported.

@note To get the class of a hash instance use @a BOM.baseClass.
*/
FALCON_DECLARE_FUNCTION(hash_r, "hash:S|Class,data:S,...")
FALCON_DEFINE_FUNCTION_P(hash_r)
{
   internal_hash( this, ctx, pCount, false );
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
FALCON_DECLARE_FUNCTION(makeHash, "name:S,silent:[B]")
FALCON_DEFINE_FUNCTION_P(makeHash)
{
   ModuleHash* mod = static_cast<ModuleHash*>(module());
   if( pCount < 1 || !ctx->param(0)->isString())
   {
       throw paramError(__LINE__, SRC);
   }
    
    String *name = ctx->param(0)->asString();
    bool silent =  pCount > 1 && ctx->param(1)->isTrue();
    Class *cls = mod->getClass(*name);

    if(!cls)
    {
        if(silent)
        {
            ctx->returnFrame();
            return;
        }

        throw paramError("Hash \"" + *name + "\" not found", __LINE__, SRC );
    }

    void* data = cls->createInstance();
    ctx->returnFrame( FALCON_GC_STORE(cls,data) );
}


static void internal_hmac( Function* caller, VMContext* ctx, bool raw )
{
   Item *i_which = ctx->param(0);
   Item *i_key = ctx->param(1);
   Item *i_data = ctx->param(2);
   if(
    !(i_which->isClass() || i_which->isString())
    || !(i_key->isMemBuf() || i_key->isString())
    || !(i_data->isMemBuf() || i_data->isString()) )
   {
       throw caller->paramError(__LINE__, SRC);
   }

   ModuleHash* mod = static_cast<ModuleHash*>(caller->module());
   Class* hashCls;
   if( i_which->isString() )
   {
      const String& name = *i_which->asString();
      hashCls = mod->getClass(name);
      if( hashCls == 0 )
      {
         throw caller->paramError("Hash \"" + name + "\" unknown", __LINE__, SRC);
      }
   }
   else {
      hashCls = static_cast<Class*>(i_which->asInst());
   }

   // first, get the 3 hash items to use
   Mod::HashBase *hash[3] = {
            static_cast<Mod::HashBase*>(hashCls->createInstance()),
            static_cast<Mod::HashBase*>(hashCls->createInstance()),
            static_cast<Mod::HashBase*>(hashCls->createInstance())
   };

   uint32 blocksize = hash[0]->GetBlockSize();
   uint32 byteCount;

   byte i_key_pad[MAX_USED_BLOCKSIZE];
   byte o_key_pad[MAX_USED_BLOCKSIZE];

   String *str = i_key->asString();
   byteCount = str->size();

   if(byteCount > blocksize) // key too large? hash it, so the resulting size will be equal to blocksize
   {
      hash[0]->UpdateData(*i_key->asString());
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
      String *str = i_key->asString();
      memptr = str->getRawStorage();
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
    hash[1]->UpdateData(*i_data->asString());
    hash[1]->Finalize();

    // outer hash
    hash[2]->UpdateData(o_key_pad, blocksize);
    hash[2]->UpdateData(hash[1]->GetDigest(), hash[1]->DigestSize());
    hash[2]->Finalize();

    uint32 size = hash[2]->DigestSize();
    byte *digest = hash[2]->GetDigest();
    String* result = new String;

    if(raw)
    {
       result->adoptMemBuf( digest, size, 0 );
       result->toMemBuf();
    }
    else
    {
       Mod::hashToString( *result, false, digest, size );
    }

    // cleanup
    for(uint32 i = 0; i < 3; ++i)
       hashCls->dispose(hash[i]);

    ctx->returnFrame( FALCON_GC_HANDLE(result) );
}

/*#
@function hmac
@brief Provides HMAC authentication for a block of data
@param which Hash that should be used
@param key Secret authentication key
@param data The data to be authenticated
@return A lowercase hexadecimal string with the HMAC-result of the chosen hash.
@raise ParamError in case @b which is a string and a hash with that name was not found; or if @b which does not evaluate to a hash object.

Param @i which can contain a String with the name of the hash, a hash class constructor, or a function that returns a usable hash object.
Unlike the hash() function, it is not possible to pass a hash object directly, because it would have to be used 3 times, which is not possible
because of finalization.

In total, this function evaluates @i which 3 times, creating 3 hash objects internally.

@note Use getSupportedHashes() to check which hash names are supported.
*/
FALCON_DECLARE_FUNCTION(hmac, "which:S|Class,key:S,data:S")
FALCON_DEFINE_FUNCTION_P1(hmac)
{
   internal_hmac(this, ctx, false);
}

/*#
@function hmac_r
@brief Provides HMAC authentication for a block of data
@param which Hash that should be used
@param key Secret authentication key
@param data The data to be authenticated
@return A raw memory buffer string containing the required hash.
@raise ParamError in case @b which is a string and a hash with that name was not found; or if @b which does not evaluate to a hash object.

Param @i which can contain a String with the name of the hash, or a hash class.

In total, this function evaluates @i which 3 times, creating 3 hash objects internally.

@note Use getSupportedHashes() to check which hash names are supported.
*/
FALCON_DECLARE_FUNCTION(hmac_r, "which:S|Class,key:S,data:S")
FALCON_DEFINE_FUNCTION_P1(hmac_r)
{
   internal_hmac(this, ctx, true);
}


//==============================================================================
// Base Hash class handler
//==============================================================================

namespace CHash {

/*#
@method update Hash
@brief Feeds data into the hash function.
@param ...
@raise AccessError if the hash is already finalized.
@raise GenericError in case of a stack overflow. This can happen with self- or circular references inside objects.
@return The object itself.

This method accepts an @i arbitrary amount of parameters, each treated differently:
- Strings are hashed with respect to their byte count (1, 2, or 4 byte strings) and endianess.
- Arrays are traversed, each item being hashed.
- Nil as parameter is always skipped, even if contained in a Sequence.- In all other cases, the parameter is converted to a string.
- To hash integer values, use @b updateInt(), as it respects their memory layout. If put into @b update(), they will be hashed as SINGLE @b BYTE.
- All parameters are hashed in the order they are passed.
- Sequences can be nested.

@note Multiple calls can be chained, e.g. hash.update(x).update(y).update(z)
*/
FALCON_DECLARE_FUNCTION(update, "...")
FALCON_DEFINE_FUNCTION_P(update)
{
    Mod::HashBase *hash = static_cast<Mod::HashBase*>(ctx->self().asInst());
    if(hash->IsFinalized())
    {
        throw new Falcon::AccessError(
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra("Already finalized hash"));
    }
    for(int32 i = 0; i < pCount; i++)
    {
        Item *what = ctx->param(i);
        Hash_updateItem_internal(what, hash, ctx, 0);
    }

    ctx->returnFrame(ctx->self());
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
FALCON_DECLARE_FUNCTION(updateInt, "...")
FALCON_DEFINE_FUNCTION_P(updateInt)
{
   Mod::HashBase *hash = static_cast<Mod::HashBase*>(ctx->self().asInst());
   if(hash->IsFinalized())
   {
        throw new Falcon::AccessError(
            Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
            .extra("Already finalized hash"));
    }
    if( pCount < 2)
    {
        throw new Falcon::ParamError(
            Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
            .extra( "N, N" ) );
    }
    uint64 num = ctx->param(0)->forceIntegerEx();
    uint8 bytes = (uint8)ctx->param(1)->forceIntegerEx();
    if( !(bytes && bytes <= 8) )
    {
        throw new Falcon::ParamError(
            Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ )
            .extra( "bytes must be in 1..8" ) );
    }
    num = endianInt64(num);
    hash->UpdateData((byte*)&num, bytes);

    ctx->returnFrame(ctx->self());
}


/*#
@method finalize Hash
@brief Finalizes the hash.
@return this same object.
*/
FALCON_DECLARE_FUNCTION(finalize, "")
FALCON_DEFINE_FUNCTION_P1(finalize)
{
    Mod::HashBase *hash = static_cast<Mod::HashBase*>(ctx->self().asInst());
    if(! hash->IsFinalized())
    {
       hash->Finalize();
    }
    ctx->returnFrame(ctx->self());
}


/*#
@property finalized HashBase
@brief Checks if a hash is finalized.

When a result from a hash is obtained, the hash will be finalized, making it impossible to add additional data.
This method can be used if the finalization state of a hash is unknown.
*/
static void get_finalized( const Class*, const String&, void* instance, Item& target)
{
    Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
    target.setBoolean(hb->IsFinalized());
}

/*#
@property bytes HashBase
@brief Returns the byte length of the hash result.

The amount of returned bytes is specific for each hash algorithm.
*/
static void get_bytes( const Class*, const String&, void* instance, Item& target)
{
    Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
    target.setInteger(hb->DigestSize());
}


/*#
@property bits HashBase
@brief Returns the bit length of the hash result.

The bit length of a hash function is a rough indicator for its safety - long hashes take exponentially longer to find a collision,
or to break them.

@note This method is a shortcut for @b bytes() * 8
*/
static void get_bits( const Class*, const String&, void* instance, Item& target)
{
    Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
    target.setInteger(hb->DigestSize()*8);
}


static void internal_toString( Function*, VMContext* ctx, bool isRaw, bool isUpper )
{
   Mod::HashBase *hash = static_cast<Mod::HashBase*>(ctx->self().asInst());

   if(!hash->IsFinalized())
   {
      hash->Finalize();
   }

   uint32 size = hash->DigestSize();
   String *buf = new String(size);
   if(byte *digest = hash->GetDigest())
   {
      if( isRaw )
      {
         buf->adoptMemBuf(digest, size, 0 );
         buf->toMemBuf();
      }
      else
      {
         Mod::hashToString(*buf, isUpper, digest, size );
      }
      ctx->returnFrame( FALCON_GC_HANDLE(buf) );
   }
   else
   {
      throw new Falcon::AccessError(
         Falcon::ErrorParam( e_acc_forbidden, __LINE__ )
         .extra("Hash could not be finalized"));
   }
}

/*#
@method toMemBuf HashBase
@brief Returns the hash result in a memory buffer string.
@return The hash result, in a 1-byte wide memory buffer string.

@note Calling this method will finalize the hash.
*/
FALCON_DECLARE_FUNCTION(toMemBuf, "")
FALCON_DEFINE_FUNCTION_P1(toMemBuf)
{
   internal_toString( this, ctx, true, false );
}


/*#
@method toString HashBase
@brief Returns the hash result as a hexadecimal string.
@return The hash result, as a lowercased hexadecimal string.

@note Calling this method will finalize the hash.
*/
FALCON_DECLARE_FUNCTION(toString, "" )
FALCON_DEFINE_FUNCTION_P1(toString)
{
   internal_toString( this, ctx, false, false );
}

/*#
@method toString HashBase
@brief Returns the hash result as a uppercase hexadecimal string.
@return The hash result, as a uppercase hexadecimal string.

@note Calling this method will finalize the hash.
*/
FALCON_DECLARE_FUNCTION(toUString, "" )
FALCON_DEFINE_FUNCTION_P1(toUString)
{
   internal_toString( this, ctx, false, true );
}


/*#
@method toInt HashBase
@brief Returns the result as an Integer value.
@return The checksum result, as an Integer.

Converts up to 8 bytes from the actual hash result to an integer value and returns it, depending on its length.
If the hash is longer, the 8 lowest bytes are taken.

@note Calling this method will finalize the hash.
@note The returned integer is in native endianness.
*/
FALCON_DECLARE_FUNCTION(toInt, "" )
FALCON_DEFINE_FUNCTION_P1(toInt)
{
   Mod::HashBase *hash = static_cast<Mod::HashBase*>(ctx->self().asInst());

   if(!hash->IsFinalized())
   {
      hash->Finalize();
   }

   ctx->returnFrame( Item().setInteger(hash->AsInt()) );
}

}

ClassHash::ClassHash():
         Class("Hash")
{
   addProperty("finalized", &CHash::get_finalized );
   addProperty("bytes", &CHash::get_bytes );
   addProperty("bits", &CHash::get_bits );

   addMethod( new CHash::Function_update );
   addMethod( new CHash::Function_updateInt );
   addMethod( new CHash::Function_toInt );
   addMethod( new CHash::Function_toString );
   addMethod( new CHash::Function_toUString );
   addMethod( new CHash::Function_toMemBuf );
   addMethod( new CHash::Function_finalize );
}

ClassHash::ClassHash( const String& name, Class* parent ):
         Class(name)
{
   setParent(parent);
}

ClassHash::~ClassHash()
{}


bool ClassHash::op_init( VMContext*, void*, int32 ) const
{
   // nothing to initialize
   return false;
}

void* ClassHash::createInstance() const
{
   return 0;
}

void* ClassHash::clone( void* ) const
{
   return 0;
}

void ClassHash::dispose( void* instance ) const
{
   Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
   delete hb;
}

void ClassHash::gcMarkInstance( void* instance, uint32 mark ) const
{
   Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
   hb->gcMark(mark);
}

bool ClassHash::gcCheckInstance( void* instance, uint32 mark ) const
{
   Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
   return hb->currentMark() >= mark;
}


void ClassHash::store( VMContext* , DataWriter* stream, void* instance ) const
{
   Mod::HashBase* hb = static_cast<Mod::HashBase*>(instance);
   hb->store( stream );
}


void ClassHash::restore( VMContext* ctx, DataReader* stream ) const
{
   Mod::HashBase* hb = static_cast<Mod::HashBase*>(createInstance());
   try
   {
      hb->restore(stream);
      ctx->pushData( FALCON_GC_STORE(this,hb) );
   }
   catch(...)
   {
      delete hb;
      throw;
   }
}


//==========================================================================================
// Class handlers -- as they're similar, we use macros
//==========================================================================================
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

template <class HASH> FALCON_FUNC Func_hashSimple( ::Falcon::VMContext *ctx, int32 pCount )
{
    HASH hash(0);

    for(int32 i = 0; i < pCount; i++)
    {
        Item *what = ctx->param(i);
        Hash_updateItem_internal(what, &hash, ctx, 0);
    }

    hash.Finalize();
    String* str = new String;
    Mod::hashToString(*str, false, hash.GetDigest(), hash.DigestSize());
    ctx->retval(FALCON_GC_HANDLE(str));
}


#define HASH_CLASS_HANDLER(__name__, __publishName__ ) \
         class Class##__name__: public ClassHash\
         {\
         public:\
            Class##__name__(Class* base);\
            virtual ~Class##__name__();\
            virtual void* createInstance() const;\
            virtual void* clone( void* instance ) const;\
         };\
         Class##__name__::Class##__name__( Class* base ):\
            ClassHash( #__publishName__, base )\
         {}\
         Class##__name__::~Class##__name__()\
         {}\
         void* Class##__name__::createInstance() const\
         {\
            return new Mod::__name__(this);\
         }\
         void* Class##__name__::clone( void* instance ) const\
         {\
            Mod::__name__* inst = static_cast<Mod::__name__*>(instance);\
            return new Mod::__name__(*inst);\
         }\
         template void Func_hashSimple<Mod::__name__>( ::Falcon::VMContext *ctx, int32 pCount );
/*
class ClassCRC32 : public ClassHash
{
public:
   ClassCRC32(Class* base);
   virtual ~ClassCRC32();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};

ClassCRC32::ClassCRC32( Class* base ):
   ClassHash( "CRC32", base )
{}

void* ClassCRC32::createInstance() const
{
   return new Mod::CRC32(this);
}

void* clone( void* instance ) const
{
   Mod::CRC32* inst = static_cast<Mod::CRC32>(instance);
   return new Mod::CRC32(*inst);
}
*/

HASH_CLASS_HANDLER(CRC32, CRC32)
HASH_CLASS_HANDLER(Adler32, Adler32)

HASH_CLASS_HANDLER(SHA1Hash, SHA1)
HASH_CLASS_HANDLER(SHA224Hash, SHA224)
HASH_CLASS_HANDLER(SHA256Hash, SHA256)
HASH_CLASS_HANDLER(SHA384Hash, SHA384)
HASH_CLASS_HANDLER(SHA512Hash, SHA512)

HASH_CLASS_HANDLER(MD2Hash, MD2)
HASH_CLASS_HANDLER(MD4Hash, MD4)
HASH_CLASS_HANDLER(MD5Hash, MD5)

HASH_CLASS_HANDLER(WhirlpoolHash, Whirlpool)
HASH_CLASS_HANDLER(TigerHash, Tiger)

HASH_CLASS_HANDLER(RIPEMD128Hash, RIPEMD128)
HASH_CLASS_HANDLER(RIPEMD160Hash, RIPEMD160)
HASH_CLASS_HANDLER(RIPEMD256Hash, RIPEMD256)
HASH_CLASS_HANDLER(RIPEMD320Hash, RIPEMD320)



ModuleHash::ModuleHash():
         Module("hash")
{
   m_baseHashCls = new ClassHash;
   addMantra(m_baseHashCls);

   addMantra( new ClassCRC32(m_baseHashCls) );
   addMantra( new ClassAdler32(m_baseHashCls) );
   addMantra( new ClassSHA1Hash(m_baseHashCls) );
   addMantra( new ClassSHA224Hash(m_baseHashCls) );
   addMantra( new ClassSHA256Hash(m_baseHashCls) );
   addMantra( new ClassSHA384Hash(m_baseHashCls) );
   addMantra( new ClassSHA512Hash(m_baseHashCls) );
   addMantra( new ClassMD2Hash(m_baseHashCls) );
   addMantra( new ClassMD4Hash(m_baseHashCls) );
   addMantra( new ClassMD5Hash(m_baseHashCls) );
   addMantra( new ClassTigerHash(m_baseHashCls) );
   addMantra( new ClassWhirlpoolHash(m_baseHashCls) );
   addMantra( new ClassRIPEMD128Hash(m_baseHashCls) );
   addMantra( new ClassRIPEMD160Hash(m_baseHashCls) );
   addMantra( new ClassRIPEMD256Hash(m_baseHashCls) );
   addMantra( new ClassRIPEMD320Hash(m_baseHashCls) );

   addMantra( new ExtFunc("crc32", "", &Func_hashSimple<Mod::CRC32>, this ) );
   addMantra( new ExtFunc("adler32", "", &Func_hashSimple<Mod::Adler32>, this ) );
   addMantra( new ExtFunc("sha1", "", &Func_hashSimple<Mod::SHA1Hash>, this ) );
   addMantra( new ExtFunc("sha224", "", &Func_hashSimple<Mod::SHA224Hash>, this ) );
   addMantra( new ExtFunc("sha256", "", &Func_hashSimple<Mod::SHA256Hash>, this ) );
   addMantra( new ExtFunc("sha348", "", &Func_hashSimple<Mod::SHA384Hash>, this ) );
   addMantra( new ExtFunc("sha512", "", &Func_hashSimple<Mod::SHA512Hash>, this ) );
   addMantra( new ExtFunc("md2", "", &Func_hashSimple<Mod::MD2Hash>, this ) );
   addMantra( new ExtFunc("md4", "", &Func_hashSimple<Mod::MD4Hash>, this ) );
   addMantra( new ExtFunc("md5", "", &Func_hashSimple<Mod::MD5Hash>, this ) );
   addMantra( new ExtFunc("whirlpool", "", &Func_hashSimple<Mod::WhirlpoolHash>, this ) );
   addMantra( new ExtFunc("tiger", "", &Func_hashSimple<Mod::TigerHash>, this ) );
   addMantra( new ExtFunc("ripemd128", "", &Func_hashSimple<Mod::RIPEMD128Hash>, this ) );
   addMantra( new ExtFunc("ripemd160", "", &Func_hashSimple<Mod::RIPEMD160Hash>, this ) );
   addMantra( new ExtFunc("ripemd256", "", &Func_hashSimple<Mod::RIPEMD256Hash>, this ) );
   addMantra( new ExtFunc("ripemd320", "", &Func_hashSimple<Mod::RIPEMD320Hash>, this ) );

   addMantra( new ExtFunc("getSupportedHashes", "", &Func_GetSupportedHashes, this ) );
}


ModuleHash::~ModuleHash()
{
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


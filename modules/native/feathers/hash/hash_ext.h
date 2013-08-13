/*
   FALCON - The Falcon Programming Language.
   FILE: hash_ext.h

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
   Interface extension functions - header file
*/

#ifndef FALCON_FEATHERS_HASH_EXT_H
#define FALCON_FEATHERS_HASH_EXT_H

#include <falcon/module.h>
#include <falcon/class.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC Func_GetSupportedHashes( ::Falcon::VMachine *vm );
FALCON_FUNC Func_hash( ::Falcon::VMachine *vm );
FALCON_FUNC Func_makeHash( ::Falcon::VMachine *vm );
FALCON_FUNC Func_hmac( ::Falcon::VMachine *vm );

void Hash_updateItem_internal(Item *what, Mod::HashBase *hash, ::Falcon::VMachine *vm, uint32 stackDepth);


class ModHash: public Module
{
public:
   Class* m_baseHashCls;

   ModHash();
   ~ModHash();
};


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
};


class ClassCRC32 : public ClassHash
{
public:
   ClassCRC32(Class* base);
   virtual ~ClassCRC32();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassAdler32 : public ClassHash
{
public:
   ClassAdler32(Class* base);
   virtual ~ClassAdler32();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassSHA1Hash : public ClassHash
{
public:
   ClassSHA1Hash(Class* base);
   virtual ~ClassSHA1Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassSHA224Hash : public ClassHash
{
public:
   ClassSHA224Hash(Class* base);
   virtual ~ClassSHA224Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassSHA256Hash : public ClassHash
{
public:
   ClassSHA256Hash(Class* base);
   virtual ~ClassSHA256Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassSHA384Hash : public ClassHash
{
public:
   ClassSHA384Hash(Class* base);
   virtual ~ClassSHA384Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassSHA512Hash : public ClassHash
{
public:
   ClassSHA512Hash(Class* base);
   virtual ~ClassSHA512Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassMD2Hash : public ClassHash
{
public:
   ClassMD2Hash(Class* base);
   virtual ~ClassMD2Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassMD4Hash : public ClassHash
{
public:
   ClassMD4Hash(Class* base);
   virtual ~ClassMD4Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassMD5Hash : public ClassHash
{
public:
   ClassMD5Hash(Class* base);
   virtual ~ClassMD5Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassWhirlpoolHash : public ClassHash
{
public:
   ClassWhirlpoolHash(Class* base);
   virtual ~ClassWhirlpoolHash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassTigerHash : public ClassHash
{
public:
   ClassTigerHash(Class* base);
   virtual ~ClassTigerHash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassRIPEMDHashBase : public ClassHash
{
public:
   ClassRIPEMDHashBase(Class* base);
   virtual ~ClassRIPEMDHashBase();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassRIPEMD128Hash : public ClassHash
{
public:
   ClassRIPEMD128Hash(Class* base);
   virtual ~ClassRIPEMD128Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassRIPEMD160Hash : public ClassHash
{
public:
   ClassRIPEMD160Hash(Class* base);
   virtual ~ClassRIPEMD160Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassRIPEMD256Hash : public ClassHash
{
public:
   ClassRIPEMD256Hash(Class* base);
   virtual ~ClassRIPEMD256Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};


class ClassRIPEMD320Hash : public ClassHash
{
public:
   ClassRIPEMD320Hash(Class* base);
   virtual ~ClassRIPEMD320Hash();

   virtual void* createInstance() const;
   virtual void* clone( void* instance ) const;
};

}
}

#endif

/* end of hash_ext.h */

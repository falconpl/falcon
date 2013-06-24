/*
   FALCON - The Falcon Programming Language.
   FILE: bufext_ext.cpp

   Buffering extensions
   Interface extension functions
   -------------------------------------------------------------------
   Author: Maximilian Malek
   Begin: Sun, 20 Jun 2010 18:59:55 +0200

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
   Buffering extensions
   Interface extension functions - header file
*/

#ifndef bufext_ext_H
#define bufext_ext_H

#include <falcon/types.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/classes/classerror.h>
#include <falcon/class.h>

#include "bufext_st.h"
#include "bytebuf.h"
#include "bitbuf.h"

namespace Falcon { namespace Ext {

template <typename BUFTYPE> inline BUFTYPE& vmGetBuf( ::Falcon::VMContext *ctx )
{
    return *static_cast<BUFTYPE*>(ctx->self().asInst());
}

String *ByteArrayToHex(byte *arr, uint32 size);

FALCON_FUNC BitBuf_bitCount( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_readBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_writeBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_sizeBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_rposBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_wposBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_readableBits( ::Falcon::VMContext *ctx, int pCount );
FALCON_FUNC BitBuf_bits_req( ::Falcon::VMContext *ctx, int pCount );

FALCON_FUNC BufferError_init( ::Falcon::VMContext *ctx, int pCount );

//================================================================
//

template<ByteBufEndianMode ENDIANMODE>
class ClassByteBufBase: public Class
{
public:
   ClassByteBufBase( Class* parent, const String& name ): Class(name) { if (parent != 0 ) setParent(parent); }
   virtual ~ClassByteBufBase() {}

   virtual void dispose( void* instance ) const { delete static_cast< ByteBufTemplate<ENDIANMODE>* >(instance); }
   virtual void* clone( void* instance ) const { return static_cast<ByteBufTemplate<ENDIANMODE>*>(instance)->clone(); }
   virtual void* createInstance() const { return new ByteBufTemplate<ENDIANMODE>;}
};

typedef ClassByteBufBase<ENDIANMODE_MANUAL> ClassByteBufManual;
typedef ClassByteBufBase<ENDIANMODE_NATIVE> ClassByteBufNativeEndian;
typedef ClassByteBufBase<ENDIANMODE_LITTLE> ClassByteBufLittleEndian;
typedef ClassByteBufBase<ENDIANMODE_BIG> ClassByteBufBigEndian;
typedef ClassByteBufBase<ENDIANMODE_REVERSE> ClassByteBufReverseEndian;

//================================================================
//

class BufferError: public Falcon::Error
{
public:
   BufferError( const Class* handler ): Error( handler ) {}
   BufferError( Class* handler, const ErrorParam &params ): Error( handler, params ) {}
   virtual ~BufferError() {}
};


class ClassBufferError: public ClassError
{
public:
   ClassBufferError(): ClassError("BufferError") {}
   virtual ~ClassBufferError() {}
   virtual void* createInstance() const { return new BufferError(this); }
};


}} // namespace Falcon::Ext


#endif

/* end of bufext_ext.h */

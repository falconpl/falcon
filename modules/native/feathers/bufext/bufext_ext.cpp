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
   Interface extension functions
*/

#include <falcon/engine.h>
#include "bufext_ext.h"
#include "bufext_st.h"

namespace Falcon { namespace Ext {


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

FALCON_FUNC BitBuf_bitCount( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    if(Item *p0 = vm->param(0))
    {
        if(uint32 bc = (uint32)p0->forceIntegerEx())
            buf.bitcount(bc);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.bitcount());
    }
}

FALCON_FUNC BitBuf_writeBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    for(uint32 i = 0; i < uint32(vm->paramCount()); i++)
    {
        Item *itm = vm->param(i);
        buf << itm->forceInteger(); // << operator makes it append with chosen bit size
    }

    vm->retval(vm->self());
}

FALCON_FUNC BitBuf_readBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    int64 val;
    buf >> val; // >> operator makes it read with chosen bit size

    vm->retval(val);
}

FALCON_FUNC BitBuf_sizeBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    vm->retval((int64)buf.size_bits());
}

FALCON_FUNC BitBuf_rposBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    if(Item *p0 = vm->param(0))
    {
        if(uint32 bc = (uint32)p0->forceIntegerEx())
            buf.rpos_bits(bc);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.rpos_bits());
    }
}

FALCON_FUNC BitBuf_wposBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    if(Item *p0 = vm->param(0))
    {
        if(uint32 bc = (uint32)p0->forceIntegerEx())
            buf.wpos_bits(bc);
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.wpos_bits());
    }
}

FALCON_FUNC BitBuf_readableBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    vm->retval(int64(buf.size_bits() - buf.rpos_bits()));
}

FALCON_FUNC BitBuf_bits_req( ::Falcon::VMachine *vm )
{
    if(!vm->paramCount())
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( "I" ) );
    }
    int64 i = vm->param(0)->forceIntegerEx();
    if(i < 0)
        vm->retval(int64(8));
    else
        vm->retval(int64(BitBuf::bits_req(uint64(i))));
}

FALCON_FUNC BufferError_init( ::Falcon::VMachine *vm )
{
    CoreObject *obj = vm->self().asObject();
    if(!obj->getUserData())
        obj->setUserData( new BufferError );

    ::Falcon::core::Error_init( vm );
}


}} // namespace Falcon::Ext

/* end of bufext_mod.cpp */

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

/*#
@method bitCount BitBuf
@brief Sets or gets the bit count used in writeBits() and readBits()
@optparam bits The amount of bits to use
@return The BitBuf itself if @i bits was set, otherwise the amount of bits used in the bit-precise read/write functions.

Default is 8, if not explicitly set.
A bit count of 0 will not write anything, and read operations will always return 0.
Values > 64 are not recommended to use as they make no sense.

@code
    bb = BitBuf()
    bb.bitCount(3).writeBits(7,4,3).bitCount(5).writeBits(30,20,10) // write with variable bit sizes
@endcode
*/
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

/*#
@method writeBits BitBuf
@brief Writes integers with a fixed bit width
@optparam ints An arbitrary amount of integers
@return The BitBuf itself.

This method writes the lowest @i n bits of the supplied integers, where @i n = bitCount().
Be sure to choose enough bits, otherwise the integers will be truncated.
To get the required amount of bits for an integer, use BitBuf.bitsForInt().

For numbers < 0, the sign is skipped and the number will be written as: (bitCount() + 1) - (abs(N) % bitCount()), where N is our negative number.
To restore a negative number, use readBits(true).

@note bitCount() >= 64 is the only mode that supports writing negative numbers *without* having to use readBits(true).

@code
    bb = BitBuf()
    bb.bitCount(3).writeBits(7,4,3).bitCount(5).writeBits(30,20,10) // write with variable bit sizes
@endcode
*/
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

/*#
@method readBits BitBuf
@brief Reads integers with a fixed bit width
@optparam neg An arbitrary amount of integers
@return An integer value.

Reads one integer of @i bitCount() bits from the buffer.
The returned integer is always positive, except if bitCount() is 64 and a negative number was written.
If @i is true, this method will set all upper missing bits to make the number negative,
to restore a previously written negative number.

@code
    bb = BitBuf().bitCount(3).writeBits(7,8,9)
    x = bb.readBits()
    y = bb.readBits()
    z = bb.readBits()
    // result: x == 7, y == 0, z == 1  (only the lowest 3 bits were written!)
@endcode

@note Unlike r8()..r64(), the boolean parameter for this method *forces* a negative number.
*/
FALCON_FUNC BitBuf_readBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    int64 val;
    buf >> val; // >> operator makes it read with chosen bit size

    if(vm->paramCount() && vm->param(0)->isTrue())
        val |= (uint64(-1) << buf.bitcount());

    vm->retval(val);
}

/*#
@method sizeBits BitBuf
@brief Returns the buffer size, in bits
@return The buffer size, in bits

This function returns or sets the BitBuf size precisely, which can be calculated as
(size() * 8) + X, where X is in [0...7].
*/
FALCON_FUNC BitBuf_sizeBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    vm->retval((int64)buf.size_bits());
}

/*#
@method rposBits BitBuf
@brief Returns the read position, in bits
@return The read position in bits if used as getter, otherwise the buffer itself

This function returns or sets the BitBuf read position precisely, which can be calculated as
(rpos() * 8) + X, where X is in [0...7].
*/
FALCON_FUNC BitBuf_rposBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    if(Item *p0 = vm->param(0))
    {
        buf.rpos_bits((uint32)p0->forceIntegerEx());
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.rpos_bits());
    }
}

/*#
@method wposBits BitBuf
@brief Returns the write position, in bits
@return The write position in bits if used as getter, otherwise the buffer itself

This function returns the BitBuf write position precisely, which can be calculated as
(wpos() * 8) + X, where X is in [0...7].
*/
FALCON_FUNC BitBuf_wposBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);

    if(Item *p0 = vm->param(0))
    {
        buf.wpos_bits((uint32)p0->forceIntegerEx());
        vm->retval(vm->self());
    }
    else
    {
        vm->retval((int64)buf.wpos_bits());
    }
}

/*#
@method readableBits BitBuf
@brief Returns the amount of bits left that can be read
@return The remaining bits until the end of the BitBuf is reached

This function returns the remaining bits precisely, which can be calculated as
(readable() * 8) + X, where X is in [0...7].
*/
FALCON_FUNC BitBuf_readableBits( ::Falcon::VMachine *vm )
{
    BitBuf& buf = vmGetBuf<BitBuf>(vm);
    vm->retval(int64(buf.size_bits() - buf.rpos_bits()));
}

/*#
@method bitsForInt BitBuf
@brief Static. Returns the amount of bits required to store an integer of the given value
@param n Integer to check
@return The amount of bits required to store an integer of the given value

Calculates how many bits are required to hold the value of the passed integer without losing data.

@note A negative number can be 1 greater then its corresponding positive number, and yield the same result (-8 needs 3 bits, where +8 needs 4, for example)
*/
FALCON_FUNC BitBuf_bits_req( ::Falcon::VMachine *vm )
{
    if(!vm->paramCount())
    {
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_mod ).extra( "I" ) );
    }
    int64 i = vm->param(0)->forceIntegerEx();
    if(i < 0)
        vm->retval(int64(BitBuf::bits_req(uint64(~(i - 1)) - 1))); // make number positive and fix 2-complement off-by-1
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

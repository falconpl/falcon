/*
   FALCON - The Falcon Programming Language.
   FILE: bufext_ext.cpp

   Buffering extensions
   Main module file, providing the module object to
   the Falcon engine.
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
   Main module file, providing the module object to
   the Falcon engine.
*/

#include <falcon/module.h>
#include <falcon/class.h>
#include <falcon/ov_names.h>

#include "bufext_ext.h"
#include "bufext_ext.inl"
#include "bufext_st.h"

#include "../include/version.h"

#include "bytebuf.h"
#include "bitbuf.h"


/*#
@module feathers.bufext bufext
@brief Flexible memory buffers

This module provides classes that are more dynamic then the MemBuf class used in the core module,
and which are specialized to easily handle primitive datatypes and binary conversion.

@beginmodule feathers.bufext
*/

/*#
@group membufs Memory Buffers
@brief Easy memory management

Classes specialized for memory modification and binary mangling.

They share a common interface, with slightly different behavior regarding endian conversion for each class.
The BitBuf offers more functions, but it slower, and possibly only needed for special cases.
*/

/*#
@class ByteBuf
@brief Flexible memory buffer optimized for binary mangling
@optparam sizeOrObject
@optparam extrasizeOrAdopt
@ingroup membufs
@prop NATIVE_ENDIAN
@prop LITTLE_ENDIAN
@prop BIG_ENDIAN
@prop REVERSE_ENDIAN

A ByteBuf is a growable memory buffer with methods to read and write primitive datatypes and strings.
It supports streaming data in and out as well as random access.

For convenience, it can be converted to a @b MemBuf, at your option with zero cost.

Endianness can optionally be changed, by default it stores data in native endian of the host machine.

The constructor takes the following parameters:
- Nothing: Initializes an empty buffer with default settings
- Number @i n: Initializes an empty buffer with @i n bytes preallocated

- MemBuf/ByteBuf: Copies the buffer
- Any other Object type: calls toMemBuf() and copies the returned buffer's memory. If the object does not have toMemBuf(), a BufferError is raised.
- Any object type + a number @i n: copies the object, and allocates n extra bytes at the end of the buffer
- Any object type + boolean true: Adopts the object's memory, but does not copy. Use with care!

Anything else passed into the constructor will raise a ParamError.
Note that the constructor does not set/copy read or write positions or anything else.

Example code for basic usage:
@code
    bb = ByteBuf(30)                    // create a ByteBuf with an initial capacity of 30 bytes
    bb.setEndian(ByteBuf.LITTLE_ENDIAN) // we store everything as little endian
    bb.w16(0, 42, -16)                  // append 2 uint16 and 1 int16
    bb.wf(0.5)                          // append a float
    bb.write("Hello world")             // append a string, char size 1
    // .. write more data ..
    // now final buffer size is known 
    bb.setEndian(ByteBuf.BIG_ENDIAN)    // the next written bytes are stored in network byte order
    bb.wpos(0).w16(bb.size())           // seek back to start and put total size there, in big endian
    mb = bb.toMemBuf()
    // -- encrypt everything except the first 2 bytes and send via network --
    
    ....
    // -- receive membuf on the other side --
    bb = ByteBuf(mb, true)              // wrap a ByteBuf around the MemBuf, without copying occupied memory
    bb.setEndian(ByteBuf.BIG_ENDIAN)    // read in network byte order
    size = bb.r16()                     // read total size
    bb.setEndian(ByteBuf.LITTLE_ENDIAN) // rest of the buffer is in little endian
    // -- decrypt remaining (size - 2) bytes --
    a = bb.r16()                        // = 42
    b = bb.r16(true)                    // (this is a signed short) = -16
    f = bb.rf()                         // = ~ 0.5 (maybe slight precision loss)
    s = bb.readString()                 // string is null terminated, and char size 1  
    // .. read remaining data ..
@endcode
*/

/*#
@class BitBuf
@ingroup membufs
@from ByteBuf
@brief Flexible memory buffer optimized for bit-precise binary mangling

The BitBuf is basically a ByteBuf, but with special read/write/seek functions whose bit-width can be changed.
This is especially useful if a series of booleans, or integers whose maximum value is known, should be stored in a buffer,
and memory usage must be as efficient as possible (e.g. to save bytes in network packets).

Endianness is always native, and attempting to change it has no effect, thus, to prevent unexpected behavior, calling @b setEndian() raises an error.

@note The BitBuf reads and writes booleans always as one single bit.
@note Unlike the ByteBuf, the []-accessor takes the index of a bit, and returns a boolean value.
*/

/*#
@class ByteBufNativeEndian
@from ByteBuf
@ingroup membufs
@brief A specialized ByteBuf that stores data in native endian.

This ByteBuf always stores data in the host machine's @b native @b endian.

Attempting to change it has no effect, thus, to prevent unexpected behavior, calling @b setEndian() raises an error.

Note: This ByteBuf should be slightly faster then the others because of zero conversion and endian checking overhead.
(However, this will only be noticed if used directly from C++)
*/

/*#
@class ByteBufReverseEndian
@from ByteBuf
@ingroup membufs
@brief A specialized ByteBuf that stores data in reversed endian.

This ByteBuf always stores data in he host machine's @b opposite @b endian (e.g. On a little endian machine it stores as big endian, and vice versa)

Attempting to change it has no effect, thus, to prevent unexpected behavior, calling @b setEndian() raises an error.
*/

/*#
@class ByteBufLittleEndian
@from ByteBuf
@ingroup membufs
@brief A specialized ByteBuf that stores data in little endian.

This ByteBuf always stores data in @b little @b endian.

Attempting to change it has no effect, thus, to prevent unexpected behavior, calling @b setEndian() raises an error.
*/

/*#
@class ByteBufBigEndian
@from ByteBuf
@ingroup membufs
@brief A specialized ByteBuf that stores data in big endian.

This ByteBuf always stores data in @b big @b endian.

Attempting to change it has no effect, thus, to prevent unexpected behavior, calling @b setEndian() raises an error.
*/

/*#
@class BufferError
@brief Error generated by buffer I/O related failures.
@optparam code A numeric error code.
@optparam description A textual description of the error code.
@optparam extra A descriptive message explaining the error conditions.
@from Error code, description, extra

See the Error class in the core module.
*/

template <typename BUFTYPE> Falcon::Class *SimpleRegisterBuf(Falcon::Module *self, Falcon::Class *cls)
{
    cls->setConstuctor(  Falcon::Ext::Buf_init<BUFTYPE>, Falcon::String(name)+"._init" );

    cls->addMethod(cls, OVERRIDE_OP_GETINDEX, Falcon::Ext::Buf_getIndex<BUFTYPE>);
    cls->addMethod(cls, OVERRIDE_OP_SETINDEX, Falcon::Ext::Buf_setIndex<BUFTYPE>);
    cls->addMethod(cls, "setEndian", Falcon::Ext::Buf_setEndian<BUFTYPE>);
    cls->addMethod(cls, "getEndian", Falcon::Ext::Buf_getEndian<BUFTYPE>);
    cls->addMethod(cls, "size", Falcon::Ext::Buf_size<BUFTYPE>);
    cls->addMethod(cls, "resize", Falcon::Ext::Buf_resize<BUFTYPE>);
    cls->addMethod(cls, "reserve", Falcon::Ext::Buf_reserve<BUFTYPE>);
    cls->addMethod(cls, "capacity", Falcon::Ext::Buf_capacity<BUFTYPE>);
    cls->addMethod(cls, "readable", Falcon::Ext::Buf_readable<BUFTYPE>);
    cls->addMethod(cls, "growable", Falcon::Ext::Buf_growable<BUFTYPE>);
    cls->addMethod(cls, "wpos", Falcon::Ext::Buf_wpos<BUFTYPE>);
    cls->addMethod(cls, "rpos", Falcon::Ext::Buf_rpos<BUFTYPE>);
    cls->addMethod(cls, "reset", Falcon::Ext::Buf_reset<BUFTYPE>);
    cls->addMethod(cls, "write", Falcon::Ext::Buf_write<BUFTYPE, true>);
    cls->addMethod(cls, "writeNoNT", Falcon::Ext::Buf_write<BUFTYPE, false>);
    cls->addMethod(cls, "writePtr", Falcon::Ext::Buf_writePtr<BUFTYPE>)
        ->addParam("src")->addParam("bytes");
    cls->addMethod(cls, "readString", Falcon::Ext::Buf_readString<BUFTYPE>)
        ->addParam("charSize");
    cls->addMethod(cls, "readToBuf", Falcon::Ext::Buf_readToBuf<BUFTYPE>)
        ->addParam("bytes");
    cls->addMethod(cls, "readPtr", Falcon::Ext::Buf_readPtr<BUFTYPE>)
        ->addParam("dest")->addParam("bytes");
    cls->addMethod(cls, "toMemBuf", Falcon::Ext::Buf_toMemBuf<BUFTYPE>);
    cls->addMethod(cls, "ptr", Falcon::Ext::Buf_ptr<BUFTYPE>);
    cls->addMethod(cls, "toString", Falcon::Ext::Buf_toString<BUFTYPE>);

    cls->addMethod(cls, "wb", Falcon::Ext::Buf_wb<BUFTYPE>);
    cls->addMethod(cls, "w8", Falcon::Ext::Buf_w8<BUFTYPE>);
    cls->addMethod(cls, "w16", Falcon::Ext::Buf_w16<BUFTYPE>);
    cls->addMethod(cls, "w32", Falcon::Ext::Buf_w32<BUFTYPE>);
    cls->addMethod(cls, "w64", Falcon::Ext::Buf_w64<BUFTYPE>);
    cls->addMethod(cls, "wf", Falcon::Ext::Buf_wf<BUFTYPE>);
    cls->addMethod(cls, "wd", Falcon::Ext::Buf_wd<BUFTYPE>);
    cls->addMethod(cls, "rb", Falcon::Ext::Buf_rb<BUFTYPE>);
    cls->addMethod(cls, "r8", Falcon::Ext::Buf_r8<BUFTYPE>);
    cls->addMethod(cls, "r16", Falcon::Ext::Buf_r16<BUFTYPE>);
    cls->addMethod(cls, "r32", Falcon::Ext::Buf_r32<BUFTYPE>);
    cls->addMethod(cls, "r64", Falcon::Ext::Buf_r64<BUFTYPE>);
    cls->addMethod(cls, "rf", Falcon::Ext::Buf_rf<BUFTYPE>);
    cls->addMethod(cls, "rd", Falcon::Ext::Buf_rd<BUFTYPE>);
    return cls;
}

Falcon::Module *bufext_module_init(void)
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "bufext" );
   //self->engineVersion( FALCON_VERSION_NUM );
   //self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // API declarations
   //
   Falcon::Class *baseSym = new ClassByteBuf();

   SimpleRegisterBuf<Falcon::ByteBuf>(self, baseSym);
   baseSym->addConstant( "NATIVE_ENDIAN", (Falcon::int64)Falcon::ENDIANMODE_NATIVE );
   baseSym->addConstant( "LITTLE_ENDIAN", (Falcon::int64)Falcon::ENDIANMODE_LITTLE );
   baseSym->addConstant( "BIG_ENDIAN",    (Falcon::int64)Falcon::ENDIANMODE_BIG );
   baseSym->addConstant( "REVERSE_ENDIAN",(Falcon::int64)Falcon::ENDIANMODE_REVERSE );

   Falcon::Class *cls = new ClassByteBufNativeEndian(baseSym);
   SimpleRegisterBuf<Falcon::ByteBufNativeEndian>  (self, cls);
   cls = new ClassByteBufLittleEndian(baseSym);
   SimpleRegisterBuf<Falcon::ByteBufLittleEndian>  (self, "ByteBufLittleEndian", cls);
   cls = new ClassByteBufBigEndian(baseSym);
   SimpleRegisterBuf<Falcon::ByteBufBigEndian>     (self, "ByteBufBigEndian"   , cls);
   cls = new ClassByteByteBufReverseEndian(baseSym);
   SimpleRegisterBuf<Falcon::ByteBufReverseEndian> (self, "ByteBufReverseEndian", cls);

   Falcon::Class *bitcls = new ClassBitBuf(baseSym);
   SimpleRegisterBuf<Falcon::BitBuf>(self, bitcls);
   bitcls->addMethod( "bitCount", Falcon::Ext::BitBuf_bitCount, "bits:[N]" );
   bitcls->addMethod( "writeBits", Falcon::Ext::BitBuf_writeBits, "..." );
   bitcls->addMethod( "readBits", Falcon::Ext::BitBuf_readBits, "neg:[B]" );
   bitcls->addMethod( "sizeBits", Falcon::Ext::BitBuf_sizeBits, "" );
   bitcls->addMethod( "rposBits", Falcon::Ext::BitBuf_rposBits, "bits:[N]" );
   bitcls->addMethod( "wposBits", Falcon::Ext::BitBuf_wposBits, "bits:[N]" );
   bitcls->addMethod( "readableBits", Falcon::Ext::BitBuf_readableBits, "" );

   // static BitBuf methods
   bitcls->addMethod("bitsForInt", Falcon::Ext::BitBuf_bits_req, "n:[N]", true);

   self->addMantra( new ClassBufferError, true );

   return self;
}

FALCON_MODULE_DECL
{
    return bufext_module_init();
}

/* end of bufext.cpp */

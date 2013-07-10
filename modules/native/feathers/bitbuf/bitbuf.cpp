/*
   FALCON - The Falcon Programming Language.
   FILE: bitbuf.cpp

   Buffering extensions
   Bit-perfect buffer class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 08 Jul 2013 13:22:03 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

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

#include "../include/version.h"

#include "bitbuf_ext.h"
#include "bitbuf_mod.h"
#include "buffererror.h"

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

Falcon::Module *bufext_module_init(void)
{
   // initialize the module
   Falcon::Module *self = new Falcon::Module("bitbuf");

   //self->engineVersion( FALCON_VERSION_NUM );
   //self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //============================================================
   // API declarations
   //

   self->addConstant( "NATIVE_ENDIAN", (Falcon::int64)Falcon::Ext::BitBuf::e_endian_same );
   self->addConstant( "LITTLE_ENDIAN", (Falcon::int64)Falcon::Ext::BitBuf::e_endian_little );
   self->addConstant( "BIG_ENDIAN",    (Falcon::int64)Falcon::Ext::BitBuf::e_endian_big );
   self->addConstant( "REVERSE_ENDIAN",(Falcon::int64)Falcon::Ext::BitBuf::e_endian_reverse );

   Falcon::Class *bitbuf = Falcon::Ext::init_classbitbuf();

   self->addMantra( bitbuf, true );
   self->addMantra( new Falcon::Ext::ClassBitBufError, true );

   return self;
}

FALCON_MODULE_DECL
{
    return bufext_module_init();
}

/* end of bitbuf.cpp */

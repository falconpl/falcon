/*
   FALCON - The Falcon Programming Language.
   FILE: bitbuf_ext.cpp

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
   Buffering extensions
   Interface extension functions
*/

/*#
	@beginmodule feathers.bufext 
*/

#include <falcon/engine.h>
#include <falcon/stderrors.h>
#include <falcon/class.h>
#include <falcon/function.h>

#include <falcon/trace.h>

#include "bitbuf_ext.h"
#include "bitbuf_mod.h"

#include <falcon/textwriter.h>

namespace Falcon {
namespace Ext {

//============================================================================
// Class definition
//============================================================================

ClassBitBuf::ClassBitBuf():
      Class("BitBuf")
{
}

ClassBitBuf::~ClassBitBuf()
{
   // nothing to do
}

void ClassBitBuf::dispose( void* instance ) const
{
   TRACE1( "ClassBitBuf::dispose %p", instance );
   BitBuf* buffer = static_cast<BitBuf*>(instance);
   delete buffer;
}

void* ClassBitBuf::clone( void* instance ) const
{
   TRACE1( "ClassBitBuf::clone %p", instance );
   BitBuf* buffer = static_cast<BitBuf*>(instance);
   BitBuf* copy = new BitBuf( * buffer );
   return copy;
}

void* ClassBitBuf::createInstance() const
{
   MESSAGE1( "ClassBitBuf::createInstance" );
   BitBuf* copy = new BitBuf;
   return copy;
}


bool ClassBitBuf::op_init( VMContext*, void* , int32 ) const
{
   // ok as we are.
   return false;  // no need to go deep
}

void ClassBitBuf::store( VMContext* , DataWriter* stream, void* instance ) const
{
   TRACE1( "ClassBitBuf::store %p", instance );

   BitBuf* buffer = static_cast<BitBuf*>(instance);
   buffer->store( stream );
}

void ClassBitBuf::restore( VMContext* ctx, DataReader* stream ) const
{
   MESSAGE1( "ClassBitBuf::restore" );

   BitBuf* buffer = new BitBuf;
   try {
      buffer->restore( stream );
      ctx->pushData( FALCON_GC_STORE( this, buffer) );
   }
   catch( ... )
   {
      delete buffer;
      throw;
   }
}

void ClassBitBuf::gcMarkInstance( void* instance, uint32 mark ) const
{
   TRACE1( "ClassBitBuf::gcMarkInstance %p, %d", instance, mark );

   BitBuf* buffer = static_cast<BitBuf*>(instance);
   buffer->gcMark(mark);
}


bool ClassBitBuf::gcCheckInstance( void* instance, uint32 mark ) const
{
   TRACE1( "ClassBitBuf::gcCheckInstance %p, %d", instance, mark );

   BitBuf* buffer = static_cast<BitBuf*>(instance);
   return buffer->currentMark() >= mark;
}


//============================================================================
// Properties
//============================================================================


/*#
  @property len BitBuf
  @brief Size of the buffer in bits
 */
static void get_len( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->size() );
}

static int checkEndianityParam(const Item& value)
{
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( TypeError, e_param_type, .extra("N") );
   }
   int64 v = value.forceInteger();
   if(
          v != (int64) BitBuf::e_endian_big
       && v != (int64) BitBuf::e_endian_little
       && v != (int64) BitBuf::e_endian_reverse
       && v != (int64) BitBuf::e_endian_same
       )
   {
      throw FALCON_SIGN_XERROR( TypeError, e_param_range, .extra("0-4") );
   }

   return (int) v;
}


/*#
  @property rend BitBuf
  @brief Endianity setting used when reading numeric values from the buffer.
 */
static void get_rend( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->readEndianity() );
}

static void set_rend( const Class*, const String&, void* instance, const Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   int v = checkEndianityParam( value );
   buf->readEndianity( (BitBuf::t_endianity) v );
}

/*#
  @property wend BitBuf
  @brief Endianity setting used when writing numeric values to the buffer.
 */
static void get_wend( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->writeEndianity() );
}

static void set_wend( const Class*, const String&, void* instance, const Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   int v = checkEndianityParam( value );
   buf->writeEndianity( (BitBuf::t_endianity) v );
}

/*#
  @property sysend BitBuf
  @brief (static) System endianity as detected on the host system.

  Will be one of the LITTLE_ENDIAN or BIG_ENDIAN constants declared
  in this module.
 */

static void get_sysend( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->writeEndianity() );
}

//============================================================================
// Methods
//============================================================================

namespace CBitBuf {

/*#
@method write BitBuf
@brief Writes data
@param data The data to be written (will be interpreted as a sequence of bytes).
@optparam bitSize Bits to be written; defaults to the count of bits in the string.
@optparam bitStart Bit from where to start.
@return The BitBuf itself.

@code
    bb = BitBuf()
    bb.bitCount(3).write("abc",12).w16(5,2,3).write(m{0xF0},4) // write with variable bit sizes
@endcode
*/
FALCON_DECLARE_FUNCTION( write, "data:S,bitSize:[N],bitStart:[N]");
FALCON_DEFINE_FUNCTION_P1( write )
{
   Item* i_source = ctx->param(0);
   Item* i_bitSize = ctx->param(1);
   Item* i_bitStart = ctx->param(2);

   if( i_source == 0 || ! i_source->isString()
       || ( i_bitSize != 0 && ! i_bitSize->isOrdinal())
       || ( i_bitStart != 0 && ! i_bitStart->isOrdinal())
       )
   {
      throw paramError( __LINE__, SRC );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   String* source = i_source->asString();

   if( i_bitSize == 0 && i_bitStart == 0 )
   {
      buf->writeBytes( source->getRawStorage(), source->size() );
   }
   else
   {
      uint32 maxSize = source->size() * 8;
      uint32 bitSize = static_cast<uint32>( i_bitSize == 0 ? maxSize: i_bitSize->forceInteger() );
      uint32 bitStart = static_cast<uint32>( i_bitStart == 0 ? 0 : i_bitStart->forceInteger() );

      // sanitize input
      if( bitStart < maxSize )
      {
         if( bitSize + bitStart > maxSize )
         {
            bitSize = maxSize - bitStart;
         }
         buf->writeBits( source->getRawStorage(), bitSize, bitStart );
      }
   }

   ctx->returnFrame( ctx->self() );
}


/*#
@method read BitBuf
@brief Reads bits from the buffer in a target string.
@param target A target string (possibly pre-allocated)
@optparam count The count of bits to be read
@return The count of bits actually read (will be less than count if there wasn't enough bits left).
*/

FALCON_DECLARE_FUNCTION( read, "data:S,bitSize:[N],bitStart:[N]");
FALCON_DEFINE_FUNCTION_P1( read )
{
   Item* i_source = ctx->param(0);
   Item* i_bitSize = ctx->param(1);

   if( i_source == 0 || ! i_source->isString()
       || ( i_bitSize != 0 && ! i_bitSize->isOrdinal())
       )
   {
      throw paramError( __LINE__, SRC );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   String* source = i_source->asString();
   int64 size;
   if( i_bitSize != 0 )
   {
      uint64 bitSize = i_bitSize->forceInteger();
      size = bitSize;
   }
   else
   {
      size = source->size() * 8;
   }

   source->reserve(size);
   uint32 count = buf->readBits( source->getRawStorage(), size );
   source->size(count);
   ctx->returnFrame((int64) count);
}


/*#
@method toString BitBuf
@brief Converts the bit stream in a readable bit-oriented representation.
@optparam Character used for 'on' bits. Defaults to '1'.
@optparam Character used for 'off' bits. Defaults to '0'.
@return A new mutable string containing a sequence of on/off characters.

*/

FALCON_DECLARE_FUNCTION( toString, "chrOn:[S|N], chrOff:[S|N]");
FALCON_DEFINE_FUNCTION_P1( toString )
{
   Item* i_chrOn = ctx->param(0);
   Item* i_chrOff = ctx->param(1);

   if( ( i_chrOn != 0 && ! (i_chrOn->isOrdinal()|| i_chrOn->isString()) )
       || ( i_chrOff != 0 && ! (i_chrOff->isOrdinal()|| i_chrOff->isString()) )
       )
   {
      throw paramError( __LINE__, SRC );
   }

   char_t chrOn = '1';
   char_t chrOff = '0';

   if( i_chrOn != 0 )
   {
      if(i_chrOn->isOrdinal())
      {
         chrOn = (uint32) i_chrOn->asInteger();
      }
      else if( i_chrOn->asString()->size() != 0 )
      {
         chrOn = i_chrOn->asString()->getCharAt(0);
      }
   }

   if( i_chrOff != 0 )
   {
      if(i_chrOff->isOrdinal())
      {
         chrOff = (uint32) i_chrOff->asInteger();
      }
      else if( i_chrOff->asString()->size() != 0 )
      {
         chrOff = i_chrOff->asString()->getCharAt(0);
      }
   }

   String* ret = new String();
   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   buf->toString(*ret, chrOn, chrOff );

   ctx->returnFrame( FALCON_GC_HANDLE(ret) );
}


/*#
@method sizeBits BitBuf
@brief Returns the buffer size, in bits
@return The buffer size, in bits

This function returns or sets the BitBuf size precisely, which can be calculated as
(size() * 8) + X, where X is in [0...7].
*/


/*#
@method rposBits BitBuf
@brief Returns the read position, in bits
@return The read position in bits if used as getter, otherwise the buffer itself

This function returns or sets the BitBuf read position precisely, which can be calculated as
(rpos() * 8) + X, where X is in [0...7].
*/


/*#
@method wposBits BitBuf
@brief Returns the write position, in bits
@return The write position in bits if used as getter, otherwise the buffer itself

This function returns the BitBuf write position precisely, which can be calculated as
(wpos() * 8) + X, where X is in [0...7].
*/


/*#
@method readableBits BitBuf
@brief Returns the amount of bits left that can be read
@return The remaining bits until the end of the BitBuf is reached

This function returns the remaining bits precisely, which can be calculated as
(readable() * 8) + X, where X is in [0...7].
*/


/*#
@method bitsForInt BitBuf
@brief Static. Returns the amount of bits required to store an integer of the given value
@param n Integer to check
@return The amount of bits required to store an integer of the given value

Calculates how many bits are required to hold the value of the passed integer without losing data.

@note A negative number can be 1 greater then its corresponding positive number, and yield the same result (-8 needs 3 bits, where +8 needs 4, for example)
*/
}

//===============================================================
// Class creation/initialization
//===============================================================

Class* init_classbitbuf()
{
   Class* bitbuf = new ClassBitBuf;

   bitbuf->addProperty( "len", get_len );
   bitbuf->addProperty( "rend", get_rend, set_rend );
   bitbuf->addProperty( "wend", get_wend, set_wend );
   bitbuf->addProperty( "sysend", get_sysend, 0, true, false );

   bitbuf->addMethod( new CBitBuf::Function_write );
   bitbuf->addMethod( new CBitBuf::Function_read );
   bitbuf->addMethod( new CBitBuf::Function_toString );

   return bitbuf;
}

}} // namespace Falcon::Ext

/* end of bufext_ext.cpp */

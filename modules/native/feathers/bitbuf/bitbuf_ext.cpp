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

/*#
  @property avail BitBuf
  @brief Number of bits still readable in the buffer.
 */

static void get_avail( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->readable() );
}

/*#
  @property eof BitBuf
  @brief True if the read pointer is past the end of the stream.
 */

static void get_eof( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setBoolean( buf->eof() );
}


/*#
  @property wpos BitBuf
  @brief Position (in bits from the start) of the write pointer.

  Can be changed to an arbitrary value. Changing the value will cause
  the internal buffer to be consolidated. If changed to an out of range
  value, a RangeError will be raised.

  \note a negative position will move the pointer to a position relative
  to the end of the bit buffer.
 */
static void get_wpos( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->wpos() );
}

static void set_wpos( const Class*, const String&, void* instance, const Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);

   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( TypeError, e_param_type, .extra("N") );
   }

   int64 pos = value.forceInteger();
   if( pos < 0 )
   {
      pos = buf->size()+pos;
   }

   if( pos < 0 || pos > buf->size() )
   {
      throw FALCON_SIGN_ERROR( RangeError, e_param_type );
   }

   buf->wpos( (uint32) pos );
}


/*#
  @property rpos BitBuf
  @brief Position (in bits from the start) of the read pointer.

  Can be changed to an arbitrary value. Changing the value will cause
  the internal buffer to be consolidated. If changed to an out of range
  value, a RangeError will be raised.

  \note a negative position will move the pointer to a position relative
  to the end of the bit buffer.
 */
static void get_rpos( const Class*, const String&, void* instance, Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);
   value.setInteger( buf->rpos() );
}

static void set_rpos( const Class*, const String&, void* instance, const Item& value )
{
   BitBuf* buf = static_cast<BitBuf*>(instance);

   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR( TypeError, e_param_type, .extra("N") );
   }

   int64 pos = value.forceInteger();
   if( pos < 0 )
   {
      pos = buf->size()+pos;
   }

   if( pos < 0 || pos > buf->size() )
   {
      throw FALCON_SIGN_ERROR( RangeError, e_param_type );
   }

   buf->rpos( pos );
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
    bb.write("abc",12).w16(5).write(m{0xF0},4)
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
      int64 bitSize = i_bitSize == 0 ? maxSize: i_bitSize->forceInteger();
      uint32 bitStart = static_cast<uint32>( i_bitStart == 0 ? 0 : i_bitStart->forceInteger() );

      // sanitize input
      if( bitStart < maxSize )
      {
         if( bitSize < 0 )
         {
            bitSize = maxSize;
         }

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
@brief Reads bits from the buffer in a target string.io
@param target A target string (mutable)
@optparam count The count of bits to be read
@return The count of bits actually read (will be less than count if there wasn't enough bits left).
*/

FALCON_DECLARE_FUNCTION( read, "data:S,count:[N]");
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

   String* source = i_source->asString();
   if( source->isImmutable() )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_type, .extra("Output string is immutable"));
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
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
   source->size( count % 8 == 0 ? (count/8) : (count/8+1) );
   ctx->returnFrame((int64) count);
}



/*#
@method grab BitBuf
@brief Reads bits from the buffer returning them in a newly allocated string buffer.
@param count The count of bits to be read
@return The newly allocated string, or nil if the stream is at eof.
*/

FALCON_DECLARE_FUNCTION( grab, "count:[N]");
FALCON_DEFINE_FUNCTION_P1( grab )
{
   Item* i_bitSize = ctx->param(0);

   if( i_bitSize == 0 || ! i_bitSize->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   if( buf->eof() )
   {
      ctx->returnFrame();
      return;
   }

   int64 size = i_bitSize->forceInteger();

   String* source = new String;
   source->reserve(size);
   uint32 count = buf->readBits( source->getRawStorage(), size );
   source->size( count % 8 == 0 ? (count/8) : (count/8+1) );
   ctx->returnFrame(FALCON_GC_HANDLE( source ));
}


/*#
@method wbit BitBuf
@brief Writes a single bit to the bit buffer.
@param bit A number, any non-zero number will be interpreted as 1.
@return This same object.
*/

FALCON_DECLARE_FUNCTION( wbit, "bit:[N]");
FALCON_DEFINE_FUNCTION_P1( wbit )
{
   Item* i_bit = ctx->param(0);

   if( i_bit == 0 || ! i_bit->isOrdinal() )
   {
      throw paramError( __LINE__, SRC );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   int64 bit = i_bit->forceInteger();
   buf->writeBit( bit != 0 );
   ctx->returnFrame(ctx->self());
}

/*#
@method rbit BitBuf
@brief Reads a single bit from the bit buffer.
@return 1 or 0.
@raise AccessError if reading at eof.
*/

FALCON_DECLARE_FUNCTION( rbit, "" );
FALCON_DEFINE_FUNCTION_P1( rbit )
{
   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   bool bit = false;
   if( ! buf->readBit( bit ) )
   {
      throw FALCON_SIGN_ERROR( AccessError, e_read_eof );
   }
   ctx->returnFrame( (int64) (bit?1:0) );
}

#define MAKE_WRITE_FUNC( __bits__ ) \
         FALCON_DECLARE_FUNCTION( w##__bits__, "number:[N]"); \
         FALCON_DEFINE_FUNCTION_P1( w## __bits__ ) \
         {\
            Item* i_number = ctx->param(0); \
            if( i_number == 0 || ! i_number->isOrdinal() )\
            {\
               throw paramError( __LINE__, SRC );\
            }\
            BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());\
            int64 number = i_number->forceInteger();\
            buf->write##__bits__( number );\
            ctx->returnFrame(ctx->self());\
         }

#define MAKE_READ_FUNC( __bits__ ) \
         FALCON_DECLARE_FUNCTION( r##__bits__, "" );\
         FALCON_DEFINE_FUNCTION_P1( r##__bits__ )\
         {\
            BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());\
            if( buf->readable() < __bits__ )\
            {\
               throw FALCON_SIGN_ERROR( AccessError, e_read_eof );\
            }\
            ctx->returnFrame( (int64) buf->read##__bits__() );\
         }\




MAKE_WRITE_FUNC( 8 )
MAKE_WRITE_FUNC( 16 )
MAKE_WRITE_FUNC( 32 )
MAKE_WRITE_FUNC( 64 )

MAKE_READ_FUNC( 8 )
MAKE_READ_FUNC( 16 )
MAKE_READ_FUNC( 32 )
MAKE_READ_FUNC( 64 )

/*#
@method w8 BitBuf
@brief Writes a single byte to the bit buffer.
@param number A number; will be truncated so that it's in 0-255 range.
@return This same object.
*/


/*#
@method r8 BitBuf
@brief Reads a single byte from the bit buffer.
@return A number in 0-255 range.
@raise AccessError if reading at eof.
@return This same object.
*/

/*#
@method w16 BitBuf
@brief Writes a 16 bit number to the bit buffer.
@param number A number; will be truncated so that it's in 0-0xFFFF range.
@return This same object.

The bytes comprised in the number will be stored according to the @a BitBuf.wend
setting. If this setting is LITTLE_ENDIAN, or SYSTEM_ENDIAN and the system endianity
is little, the bytes will be stored in significance order (less significant last).
Otherwise, they will be stored in reverse order.

For example, if storing the number 0x0102 in little endian ordering, the resulting
writes in bit buffers would be 00000001, 00000010. Storing this number in big endian
order would result in writing the bits 00000010, 00000001.

@note The @a BitBuf.wend property defaults to LITTLE_ENDIAN.
*/

/*#
@method r16 BitBuf
@brief Reads 16 bits from the buffer and convert them to a number.
@return A number in range 0-0xFFFF.
@return This same object.
@raise AccessError if reading at eof.

The bytes composing the number are interpreted as having little or big endian
ordering depending on the setting of the @a BitBuf.rend property. The default
value for this property is LITTLE_ENDIAN.

Please, refer to @a BitBuf.w16 for a wider description.
@see BitBuf.w16
*/



/*#
@method w32 BitBuf
@brief Writes a 32 bit number to the bit buffer.
@param number A number; will be truncated so that it's in 0-0xFFFF-FFFF range.
@return This same object.

Please, refer to @a BitBuf.w16 for a wider description.
*/

/*#
@method r32 BitBuf
@brief Reads 32 bits from the buffer and convert them to a number.
@return A number in range 0-0xFFFF-FFFF.
@raise AccessError if reading at eof.

Please, refer to @a BitBuf.r16 for a wider description.
*/

/*#
@method w64 BitBuf
@brief Writes a 64 bit number to the bit buffer.
@param number A 64 bit number.
@return This same object.

In case the number is negative, the sign bit is interpreted as the 64th
bit of an unsinged number.

Please, refer to @a BitBuf.w16 for a wider description.
*/

/*#
@method r64 BitBuf
@brief Reads 64 bits from the buffer and convert them to a number.
@return A number in range 0-0xFFFF-FFFF-FFFF-FFFF.
@raise AccessError if reading at eof.

The highest bit read from the stream and placed as 64th bit is interpreted
by Falcon as a sign bit.

@note All the bitwise operations in Falcon ignore the signedness of a number
and consider the highest bit as simply the highest order bit, shifting it
around as-is. If you want to preserve the sign across a bitwise operation,
you should check the sign via a comparison, zero the highest bit and then
restore the bit after the shift is done.

Please, refer to @a BitBuf.r16 for a wider description.
*/


/*#
@method wbits BitBuf
@brief Writes up to 64 bits from a number.
@param number The number to be written.
@param bits The count of bits to be written in the buffer.
@return This same object.

This method writes an arbitrary number of least significant @b bits taken from the given
@b number into the buffer.

The endianity of the system or of the stream is not considered, as the most significant bit
among those taken from the chosen @b number is always stored first.
*/

FALCON_DECLARE_FUNCTION( wbits, "number:[N], count:[N]");
FALCON_DEFINE_FUNCTION_P1( wbits )
{
   Item* i_bit = ctx->param(0);
   Item* i_count = ctx->param(1);

   if( i_bit == 0 || ! i_bit->isOrdinal()
    || i_count == 0 || ! i_count->isOrdinal()
    )
   {
      throw paramError( __LINE__, SRC );
   }

   int64 count = i_count->forceInteger();
   if( count < 0 || count > 64 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra("0 <= count <= 64") );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   int64 bit = i_bit->forceInteger();
   buf->writeNumberBits( bit, count );
   ctx->returnFrame(ctx->self());
}

/*#
@method rbits BitBuf
@brief Reads up to 64 bits as a number.
@param bits The count of bits to be taken from the buffer.
@return The read number.

This method reads an arbitrary number of bits from the stream, and treat them
as a number, considering the first taken bit as the most significant bit of the
read number.

The endianity of the system or of the stream is not considered.

*/


FALCON_DECLARE_FUNCTION( rbits, "" );
FALCON_DEFINE_FUNCTION_P1( rbits )
{
   Item* i_count = ctx->param(0);

   if( i_count == 0 || ! i_count->isOrdinal())
   {
      throw paramError( __LINE__, SRC );
   }

   int64 count = i_count->forceInteger();
   if( count < 0 || count > 64 )
   {
      throw FALCON_SIGN_XERROR( ParamError, e_param_range, .extra("0 <= count <= 64") );
   }

   BitBuf* buf = static_cast<BitBuf*>(ctx->self().asInst());
   uint64 number;
   if( ! buf->readNumberBits( number, count ) )
   {
      throw FALCON_SIGN_ERROR( AccessError, e_read_eof );
   }
   ctx->returnFrame( (int64) number );
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
   bitbuf->addProperty( "wpos", get_wpos, set_wpos );
   bitbuf->addProperty( "rpos", get_rpos, set_rpos);
   bitbuf->addProperty( "eof", get_eof );
   bitbuf->addProperty( "avail", get_avail );
   bitbuf->addProperty( "sysend", get_sysend, 0, true, false );

   bitbuf->addMethod( new CBitBuf::Function_write );
   bitbuf->addMethod( new CBitBuf::Function_read );
   bitbuf->addMethod( new CBitBuf::Function_grab );
   bitbuf->addMethod( new CBitBuf::Function_toString );

   bitbuf->addMethod( new CBitBuf::Function_wbit );
   bitbuf->addMethod( new CBitBuf::Function_rbit );

   bitbuf->addMethod( new CBitBuf::Function_w8 );
   bitbuf->addMethod( new CBitBuf::Function_w16 );
   bitbuf->addMethod( new CBitBuf::Function_w32 );
   bitbuf->addMethod( new CBitBuf::Function_w64 );

   bitbuf->addMethod( new CBitBuf::Function_r8 );
   bitbuf->addMethod( new CBitBuf::Function_r16 );
   bitbuf->addMethod( new CBitBuf::Function_r32 );
   bitbuf->addMethod( new CBitBuf::Function_r64 );

   bitbuf->addMethod( new CBitBuf::Function_wbits );
   bitbuf->addMethod( new CBitBuf::Function_rbits );

   return bitbuf;
}

}} // namespace Falcon::Ext

/* end of bufext_ext.cpp */

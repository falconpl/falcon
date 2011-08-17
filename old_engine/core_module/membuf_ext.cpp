/*
   FALCON - The Falcon Programming Language.
   FILE: membuf_ext.cpp

   Memory buffer functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"

/*#
   @beginmodule core
*/

namespace Falcon {
namespace core {

FALCON_FUNC Make_MemBuf( ::Falcon::VMachine *vm )
{
   Item *i_size = vm->param(0);
   Item *i_wordSize = vm->param(1);

   if( ( i_size == 0 || ! i_size->isOrdinal() ) ||
       ( i_wordSize != 0 && ! i_wordSize->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime ).extra( "N,[N]" ) );
   }

   int64 wordSize = i_wordSize == 0 ? 1: i_wordSize->forceInteger();
   int64 size = i_size->forceInteger();
   if ( wordSize < 1 || wordSize > 4 || size <= 0 )
   {
      throw  new ParamError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ) );
   }

   MemBuf *mb = 0;
   switch( wordSize )
   {
      case 1: mb = new MemBuf_1( (uint32) size ); break;
      case 2: mb = new MemBuf_2( (uint32) size ); break;
      case 3: mb = new MemBuf_3( (uint32) size ); break;
      case 4: mb = new MemBuf_4( (uint32) size ); break;
   }
   fassert( mb != 0 );
   vm->retval( mb );
}

/*#
   @function MemBufFromPtr
   @ingroup memory_manipulation
   @brief Creates a memory buffer to store raw memory coming from external providers.
   @param data A pointer to the raw memory (as an integer value).
   @param size The maximum size of the memory.
   @optparam wordSize Size of each word in bytes (1, 2, 3 or 4).

   Intentionally left undocumented. Don't use if you don't know what you're doing.
*/

FALCON_FUNC Make_MemBufFromPtr( ::Falcon::VMachine *vm )
{
   Item *i_ptr = vm->param(0);
   Item *i_size = vm->param(1);
   Item *i_wordSize = vm->param(2);

   if( ( i_ptr == 0 || ! i_ptr->isInteger() ) ||
       ( i_size == 0 || ! i_size->isOrdinal() ) ||
       ( i_wordSize != 0 && ! i_wordSize->isOrdinal() )
      )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .origin( e_orig_runtime ).extra( "N,[N]" ) );
   }

   int64 wordSize = i_wordSize == 0 ? 1: i_wordSize->forceInteger();
   int64 size = i_size->forceInteger();
   if ( wordSize < 1 || wordSize > 4 || size <= 0 )
   {
      throw  new ParamError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ) );
   }

   MemBuf *mb = 0;
   byte *data = (byte*) i_ptr->asInteger();
   switch( wordSize )
   {
      case 1: mb = new MemBuf_1( data, (uint32) size, 0 ); break;
      case 2: mb = new MemBuf_2( data, (uint32) size, 0 ); break;
      case 3: mb = new MemBuf_3( data, (uint32) size, 0 ); break;
      case 4: mb = new MemBuf_4( data, (uint32) size, 0 ); break;
   }
   fassert( mb != 0 );
   vm->retval( mb );
}


/*#
   @class MemoryBuffer
   @from BOM
   @ingroup bom_classes
   @brief Metaclass for MemBuf items.

   The Memory Buffers have a set of internal pointers and sequence methods
   useful to parse binary streams read in variable size chunks.

   Initially, allocate a memory buffer wide enough to store enough data.
   The maximum possible amount of data units (generally bytes) that can
   be stored in a memory buffer is its @b length, returned by the
   BOM @b len method or @a len function.

   After having read some contents from a stream, the buffer @b limit
   will be moved to the amount of incoming data (which may be also
   the same as the @b length if the buffer is completely filled).

   The application may get one item at a time via the @a MemoryBuffer.get()
   method, or process blocks of data transferring them to other membufs
   or arrays via ranged operators.

   Each get() moves a @b position indicator forward up to the @b limit.
   At the same time, it is possible to put data in the buffer, moving
   forward the @b position pointer up to the limit.

   The buffer has a @b marker, that can be set at current @b position,
   and that can be later used to to return to a previously marked position.

   The following invariants always hold:
   @code
      0 <= [mark] <= position <= limit <= length
   @endcode

   The limit is usually set to the buffer length, unless it is explicitly set
   to a lower position via explicit calls, or the last read didn't bear
   enough data to fill the buffer.

   The following operations are meant to simplify read and partial parsing
   of binary data:
   - @a MemoryBuffer.reset: return the position at last mark (raises if a mark is not defined).
   - @a MemoryBuffer.rewind: discards the mark and returns the position to 0.
   - @a MemoryBuffer.clear: rewinds and set limit to the buffer length.
   - @a MemoryBuffer.flip: sets the limit to the current position, the position to zero and discards the mark.
   - @a MemoryBuffer.compact: removes already parsed data and prepares for an incremental read.

   All the members in this group not explicitly returning data or sizes
   return the MemPool itself, so that it is possible to concatenate calls like this:

   @code
      mb.clear()
      mb.position(3)

      // equivalent:
      mb.clear().position(3)
   @endcode
*/

/*#
   @method first MemoryBuffer
   @brief Returns an iterator to the first element of this buffer.
   @return An iterator.
*/
FALCON_FUNC MemoryBuffer_first( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   iterator->setProperty( "_pos", Item((int64) 0) );
   iterator->setProperty( "_origin", vm->self() );
   vm->retval( iterator );
}
/*#
   @method last MemoryBuffer
   @brief Returns an iterator to the last element of this buffer.
   @return An iterator.
*/

FALCON_FUNC MemoryBuffer_last( VMachine *vm )
{
   Item *itclass = vm->findWKI( "Iterator" );
   fassert( itclass != 0 );

   CoreObject *iterator = itclass->asClass()->createInstance();
   MemBuf* orig = vm->self().asMemBuf();
   iterator->setProperty( "_pos", Item(orig->length() == 0 ? 0 : (int64) orig->length() - 1) );
   iterator->setProperty( "_origin", vm->self() );
   vm->retval( iterator );
}

/*#
   @method front MemoryBuffer
   @brief Returns the first element in this memory buffer.
   @return A number representing the first element in this buffer.
   @raise AccessError if this buffer is empty.
*/
FALCON_FUNC MemoryBuffer_front( VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   if ( self->length() == 0 )
   {
       throw new AccessError( ErrorParam( e_arracc, __LINE__ ).origin( e_orig_runtime ) );
   }
   else {
       vm->retval( (int64) self->get(0) );
   }
}
/*#
   @method back MemoryBuffer
   @brief Returns the last element in this memory buffer.
   @return A number representing the last element in this buffer.
   @raise AccessError if this buffer is empty.
*/

FALCON_FUNC MemoryBuffer_back( VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   if ( self->length() == 0 )
   {
       throw new AccessError( ErrorParam( e_arracc, __LINE__ ).origin( e_orig_runtime ) );
   }
   else {
       vm->retval( (int64) self->get(self->length() - 1) );
   }
}

/*#
   @method put MemoryBuffer
   @brief Puts a value in the memory buffer.
   @return The buffer itself.
   @raise AccessError if the buffer is full (limit() == position())

   Sets the vaulue at current position and advances it.
*/

FALCON_FUNC MemoryBuffer_put( ::Falcon::VMachine *vm )
{
   Item *i_data = vm->param(0);
   if ( i_data == 0 || ! i_data->isOrdinal() )
   {
      throw  new ParamError( ErrorParam( e_inv_params, __LINE__ ).origin( e_orig_runtime ).extra( "N" ) );
      return;
   }

   MemBuf* self = vm->self().asMemBuf();
   self->put( (uint32) i_data->forceInteger() );
   vm->retval( self );
}

/*#
   @method get MemoryBuffer
   @brief Gets a value in the memory buffer.
   @return the retreived value.
   @raise AccessError if the buffer is full (limit() == position())

   Gets the vaulue at current position and advances it.
*/

FALCON_FUNC MemoryBuffer_get( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   vm->regA().setInteger( self->get() );
}

/*#
   @method rewind MemoryBuffer
   @brief Rewinds the buffer and discards the mark.
   @return The buffer itself

   Returns the position at 0 and discards the mark. Limit is unchanged.
*/
FALCON_FUNC MemoryBuffer_rewind( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->rewind();
   vm->retval( self );
}


/*#
   @method reset MemoryBuffer
   @brief Returns the position to the last mark.
   @return The buffer itself
   @raise AccessError if the mark is not defined.

   Returns the position at the value previously marked with @a MemoryBuffer.mark.
   The mark itself is not discarded.
*/
FALCON_FUNC MemoryBuffer_reset( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->reset();
   vm->retval( self );
}

/*#
   @method remaining MemoryBuffer
   @brief Determines how many items can be read.
   @return Count of remanining items.

   Returns the count of times get and put can be called on this memory buffer
   before raising an error.
*/
FALCON_FUNC MemoryBuffer_remaining( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   vm->retval( (int64) self->remaining() );
}

/*#
   @method compact MemoryBuffer
   @brief Discards processed data and prepares to a new read.
   @return The buffer itself

   This operation moves all the bytes between the position (included) and the limit
   (escluded) to the beginning of the buffer. Then, the position is moved to the
   previous value of the limit, and the limit is reset to the length of the buffer.

   As read is performed by filling the data between the current position and the limit,
   this operation allows to trying get more data when a previously imcomplete input was
   taken.

   @note Just, remember to move the position back to the first item that was still to
      be decoded before compact
*/
FALCON_FUNC MemoryBuffer_compact( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->compact();
   vm->retval( self );
}

/*#
   @method flip MemoryBuffer
   @brief Sets the limit to the current position, and the position to zero.
   @return The buffer itself

   This is useful to write some parsed or created contents back on a stream.
   Once filled the buffer with data, the position is on the last element;
   then, using flip and writing the buffer, the prepared data is sent to
   the output.
*/
FALCON_FUNC MemoryBuffer_flip( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->flip();
   vm->retval( self );
}


/*#
   @method mark MemoryBuffer
   @brief Places the mark at current position.
   @return The buffer itself

   This position sets the mark at current position; calling @a MemoryBuffer.reset later
   on will move the position back to this point.
*/
FALCON_FUNC MemoryBuffer_mark( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->placeMark();
   vm->retval( self );
}


/*#
   @method position MemoryBuffer
   @brief Sets or get the current position in the buffer.
   @optparam pos The new position
   @return The buffer itself or the requested position.
   @raise AccessError if the @b pos parameter is > limit.

   This method can be used to get the current postition in the memory buffer,
   or to set a new position. In the first case, the current value of the
   position is returned, in the second case the current buffer is returned.

   As the position can never be greater that the limit, an error is raised in case
   a value too high has been set.
*/
FALCON_FUNC MemoryBuffer_position( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   Item *i_pos = vm->param(0);
   if ( i_pos == 0 )
   {
      vm->regA().setInteger( self->position() );
   }
   else if ( ! i_pos->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "N" ) );
   }
   else
   {
      self->position( (uint32) i_pos->forceInteger() );
      vm->retval( self );
   }
}

/*#
   @method clear MemoryBuffer
   @brief Clears the buffer resetting it to initial status.
   @return The buffer itself.

   Removes the mark, sets the position to zero and the limit to the length.
   This makes the buffer ready for a new set of operations.
*/
FALCON_FUNC MemoryBuffer_clear( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->clear();
   vm->retval( self );
}


/*#
   @method limit MemoryBuffer
   @brief Gets or sets current filled data size.
   @optparam pos The value to be applied to the memory buffer.
   @return current limit.

   This method can be used to get the current postition in the memory buffer,
   or to set a new position. In the first case, the current value of the
   position is returned, in the second case the current buffer is returned.
*/
FALCON_FUNC MemoryBuffer_limit( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();

   Item *i_limit = vm->param(0);
   if ( i_limit == 0 )
   {
      vm->regA().setInteger( self->limit() );
   }
   else if ( ! i_limit->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).origin( e_orig_runtime ).extra( "N" ) );
   }
   else
   {
      self->limit( (uint32) i_limit->forceInteger() );
      vm->retval( self );
   }
}


/*#
   @method wordSize MemoryBuffer
   @brief Returns the number of bytes used to store each entry of this Memory Buffer.
   @return Size of each memory buffer entity in bytes.
*/
FALCON_FUNC MemoryBuffer_wordSize( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) vm->self().asMemBuf()->wordSize() );
}

/*#
   @method fill MemoryBuffer
   @brief Fills all the elements in the memory buffer with a given value.
   @param value The value to be applied to the memory buffer.
   @return This memory buffer.
*/

FALCON_FUNC MemoryBuffer_fill( ::Falcon::VMachine *vm )
{
   Item *i_item = vm->param(0);
   MemBuf* self = vm->self().asMemBuf();

   if ( i_item == 0 || ! i_item->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "X" ) );
   }

   uint32 value = (uint32) i_item->forceInteger();

   for( uint32 i = 0; i < self->length(); i ++ )
      self->set( i, value );

   vm->retval( self );
}

}
}

/* end of membuf_ext.cpp */

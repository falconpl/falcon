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
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N,[N]" ) ) );
      return;
   }

   int64 wordSize = i_wordSize == 0 ? 1: i_wordSize->forceInteger();
   int64 size = i_size->forceInteger();
   if ( wordSize < 1 || wordSize > 4 || size <= 0 )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_param_range ) ) );
      return;
   }

   MemBuf *mb = 0;
   switch( wordSize )
   {
      case 1: mb = new MemBuf_1( (uint32) size ); break;
      case 2: mb = new MemBuf_2( (uint32) size * 2); break;
      case 3: mb = new MemBuf_3( (uint32) size * 3); break;
      case 4: mb = new MemBuf_4( (uint32) size * 4); break;
   }
   fassert( mb != 0 );
   vm->retval( mb );
}


/*#
   @class MemoryBuffer
   @from BOM
   @brief Metaclass for MemBuf items.
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
       vm->raiseRTError( new AccessError( ErrorParam( e_arracc ) ) );
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
       vm->raiseRTError( new AccessError( ErrorParam( e_arracc ) ) );
   }
   else {
       vm->retval( (int64) self->get(self->length() - 1) );
   }
}


FALCON_FUNC MemoryBuffer_put( ::Falcon::VMachine *vm )
{
   Item *i_data = vm->param(0);
   if ( i_data == 0 || ! i_data->isOrdinal() )
   {
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N" ) ) );
      return;
   }
   
   MemBuf* self = vm->self().asMemBuf();
   self->put( (uint32) i_data->forceInteger() );
}

FALCON_FUNC MemoryBuffer_get( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   vm->regA().setInteger( self->get() );
}

FALCON_FUNC MemoryBuffer_rewind( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->rewind();
}

FALCON_FUNC MemoryBuffer_reset( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->reset();
}

FALCON_FUNC MemoryBuffer_flip( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->flip();
}

FALCON_FUNC MemoryBuffer_mark( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->placeMark();
}

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
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N" ) ) );
   }
   else 
   {
      self->limit( (uint32) i_pos->forceInteger() );
   }
}

FALCON_FUNC MemoryBuffer_clear( ::Falcon::VMachine *vm )
{
   MemBuf* self = vm->self().asMemBuf();
   self->clear();
}

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
      vm->raiseRTError( new ParamError( ErrorParam( e_inv_params ).extra( "N" ) ) );
   }
   else 
   {
      self->limit( (uint32) i_limit->forceInteger() );
   }
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
         .extra(  "X" ) );
   }
   
   uint32 value = (uint32) i_item->forceInteger();
   
   for( uint32 i = 0; i < self->length(); i ++ )
      self->set( i, value );
   
   vm->retval( self );
}

}
}

/* end of membuf_ext.cpp */

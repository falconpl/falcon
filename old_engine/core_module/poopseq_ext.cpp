/*
   FALCON - The Falcon Programming Language.
   FILE: poopcomp_ext.cpp

   Prototype oop oriented sequence interface.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 10 Aug 2009 11:19:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/coreobject.h>
#include <falcon/coredict.h>
#include <falcon/poopseq.h>
#include <falcon/garbagepointer.h>
#include "core_module.h"

namespace Falcon {
namespace core {

/*#
   @method comp Object
   @brief Appends elements to this object through a filter.
   @param source One sequence, range or callable generating items.
   @optparam filter A filtering function receiving one item at a time.
   @return This object.

   This method extracts one item at a time from the source, and calls repeatedly the
   @b append method of this object.

   Please, see the description of @a Sequence.comp.
   @see Sequence.comp
*/

FALCON_FUNC  Object_comp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "R|A|C|Sequence, [C]" ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreObject* obj = vm->self().asObject();

   Item i_gen = *vm->param(0);
   Item i_check = vm->param(1) == 0 ? Item(): *vm->param(1);

   PoopSeq* seq = new PoopSeq( vm, Item(obj) );  // may throw
   vm->pushParam( new GarbagePointer( seq ) );
   seq->comprehension_start( vm, vm->self(), i_check );
   vm->pushParam( i_gen );
}

/*#
   @method mcomp Object
   @brief Appends elements to this object through a filter.
   @param ... One or more sequences, ranges or callables generating items.
   @return This object.

   This method sends the data generated from multiple comprehension,
   to the append method of this object.

   Please, see the description of @a Sequence.comp.
   @see Sequence.mcomp
*/

FALCON_FUNC  Object_mcomp ( ::Falcon::VMachine *vm )
{
   // Save the parameters as the stack may change greatly.
   CoreObject* obj = vm->self().asObject();
   StackFrame* current = vm->currentFrame();

   PoopSeq* seq = new PoopSeq( vm, Item(obj) );  // may throw
   vm->pushParam( new GarbagePointer( seq ) );
   seq->comprehension_start( vm, vm->self(), Item() );

   for( uint32 i = 0; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}


/*#
   @method mfcomp Object
   @brief Appends elements to this object through a filter.
   @param filter A filter function receiving each element before its insertion, or nil.
   @param ... One or more sequences, ranges or callables generating items.
   @return This object.

   This method performs a filtered multiple comprehension and and calls repeatedly the
   @b append method of this object, passing the output of the filter function
   to it each time. If the filter function returns an oob(1), the step is skipped
   and the @b append method is not called.

   Please, see the description of @a Sequence.comp.
   @see Sequence.mfcomp
*/
FALCON_FUNC  Object_mfcomp ( ::Falcon::VMachine *vm )
{
   if ( vm->param(0) == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "C, ..." ) );
   }

   // Save the parameters as the stack may change greatly.
   CoreObject* obj = vm->self().asObject();
   StackFrame* current = vm->currentFrame();

   Item i_check = vm->param(0) == 0 ? Item(): *vm->param(0);

   PoopSeq* seq = new PoopSeq( vm, Item(obj) );  // may throw
   vm->pushParam( new GarbagePointer( seq ) );
   seq->comprehension_start( vm, vm->self(), i_check );

   for( uint32 i = 1; i < current->m_param_count; ++i )
   {
      vm->pushParam( current->m_params[i] );
   }
}

}
}

/* end of poopseq_ext.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: set_ext.cpp

   Implementation of the RTL Set Falcon class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 08 Aug 2009 14:55:46 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/*#
   @beginmodule core
*/

/** \file
   Implementation of the RTL List Falcon class.
*/


#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemset.h>
#include <falcon/iterator.h>
#include <falcon/vm.h>
#include "core_module.h"
#include <falcon/eng_messages.h>

namespace Falcon {
namespace core {

/*#
   @class Set
   @from Sequence ...
   @brief Storage for uniquely defined items (and ordering criterion).
   @param ... An arbitrary list of parameters.

   The Set class implements a binary tree, uniquely and orderly storing a
   set of generic Falcon items. Instances of the Set
   class can be used as parameters for the Iterator constructor, and an iterator
   can be generated for them using first() and last() BOM methods. Also,
   instances of the Set class can be used as any other sequence in for/in loops.

   Items in the set are ordered using the Falcon standard comparison algorithm;
   if they are instances of classes (or blessed dictionaries) implementing the
   compare() method, that method is used as a comparison criterion.

   If the set constructor is given some parameters, it will be initially filled
   with those items; if some of them is duplicated, only one item will be then
   found in the set.
*/
FALCON_FUNC  Set_init ( ::Falcon::VMachine *vm )
{
   ItemSet *set = new ItemSet;
   int32 pc = vm->paramCount();
   for( int32 p = 0; p < pc; p++ )
   {
      set->insert( *vm->param(p) );
   }

   set->owner( vm->self().asObject() );
   vm->self().asObject()->setUserData( set );
}


/*#
   @method insert Set
   @brief Adds an item to the set.
   @param item The item to be added.

   If an item considered equal to the added one exists, the
   previously set item is destroyed.
*/
FALCON_FUNC  Set_insert ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemSet *set = dyncast<ItemSet *>( vm->self().asObject()->getFalconData() );
   set->insert( *data );
}


/*#
   @method remove Set
   @brief Removes an item from a set.
   @param item The item to be removed.
   @return True if the item was removed, false if it wasn't found.
*/
FALCON_FUNC  Set_remove ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemSet *set = dyncast<ItemSet *>( vm->self().asObject()->getFalconData() );
   ItemSetElement* elem = set->find( *data );
   if( elem == 0 )
   {
      vm->regA().setBoolean( false );
   }
   else {
      set->erase( elem );
      vm->regA().setBoolean( true );
   }
}

/*#
   @method contains Set
   @brief Checks if a certain item is in the set.
   @param item The item to be found.
   @return True if the item is in the set, false otherwise.
*/
FALCON_FUNC  Set_contains ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemSet *set = dyncast<ItemSet *>( vm->self().asObject()->getFalconData() );
   ItemSetElement* elem = set->find( *data );
   vm->regA().setBoolean( elem != 0 );
}

/*#
   @method find Set
   @brief Checks if a certain item is in the set.
   @param item The item to be found.
   @return True if the item is in the set, false otherwise.
*/
FALCON_FUNC  Set_find ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemSet *set = dyncast<ItemSet *>( vm->self().asObject()->getFalconData() );
   ItemSetElement* elem = set->find( *data );
   if( elem != 0 )
   {
      Iterator* iter = new Iterator(set);
      set->getIteratorAt( *iter, elem );
      Item *i_iclass = vm->findWKI( "Iterator" );
      fassert( i_iclass != 0 && i_iclass->isClass() );
      CoreObject* citer = i_iclass->asClass()->createInstance();
      // we need this to declare the iterator as a falcon data
      citer->setUserData( iter );
      vm->retval( citer );
   }
   else
      vm->retnil();
}

/*#
   @method len Set
   @brief Returns the number of items stored in this set.
   @return Count of items in the set.
*/
FALCON_FUNC  Set_len( ::Falcon::VMachine *vm )
{
   ItemSet *set = dyncast<ItemSet *>( vm->self().asObject()->getFalconData() );
   vm->retval( (int64) set->size() );
}

}
}

/* end of set_ext.cpp */

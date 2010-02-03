/*
   FALCON - The Falcon Programming Language.
   FILE: list_ext.cpp

   Implementation of the RTL List Falcon class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01

   -------------------------------------------------------------------
   (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of the RTL List Falcon class.
*/


#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemlist.h>
#include <falcon/iterator.h>
#include <falcon/vm.h>
#include "core_module.h"
#include <falcon/eng_messages.h>

namespace Falcon {
namespace core {

/*#
   @class List
   @from Sequence ...
   @brief Fast growable double linked list.
   @param ... An arbitrary list of parameters.

   The list class implements a double linked list of generic Falcon items that
   has some hooking with the falcon VM. Particularly, instances of the List
   class can be used as parameters for the Iterator constructor, or an iterator
   can be generated for them using first() and last() BOM methods. Also,
   instances of the List class can be used as any other sequence in for/in loops.

   For example, the following code:
   @code
   descr = List("blue", "red", "gray", "purple")

   for color in descr
   forfirst
      >> "Grues are ", color
      continue
   end
   formiddle: >> ", ", color
   forlast: > " and ", color, "."
   end
   @endcode

   prints:
   @code
   Grues are blue, red, gray and purple.
   @endcode
*/

FALCON_FUNC  List_init ( ::Falcon::VMachine *vm )
{
   ItemList *list = new ItemList;
   int32 pc = vm->paramCount();
   for( int32 p = 0; p < pc; p++ )
   {
      list->push_back( *vm->param(p) );
   }

   list->owner( vm->self().asObject() );
   vm->self().asObject()->setUserData( list );
}


/*#
   @method push List
   @brief Appends given item to the end of the list.
   @param item The item to be pushed.
*/
FALCON_FUNC  List_push ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   list->push_back( *data );
}

/*#
   @method pop List
   @brief Removes the last item from the list (and returns it).
   @raise AccessError if the list is empty.
   @return The last item in the list.

   Removes the last item from the list (and returns it).
   If the list is empty, an access exception is raised.
*/
FALCON_FUNC  List_pop ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  //empty() is virtual
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }

   vm->retval( list->back() );
   list->pop_back();
}

/*#
   @method pushFront List
   @brief Pushes an item in front of the list.
   @param item The item to be pushed.
*/
FALCON_FUNC  List_pushFront ( ::Falcon::VMachine *vm )
{
   Item *data = vm->param( 0 );

   if( data == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") );
      return;
   }

   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   list->push_front( *data );
}

/*#
   @method popFront List
   @brief Removes the first item from the list (and returns it).
   @raise AccessError if the list is empty.
   @return The first item in the list.

   Removes the first item from the list (and returns it).
   If the list is empty, an access exception is raised.
*/
FALCON_FUNC  List_popFront ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  //empty() is virtual
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) );
      return;
   }

   vm->retval( list->front() );
   list->pop_front();
}

/*#
   @method len List
   @brief Returns the number of items stored in the Sequence.
   @return Count of items in the Sequence.
*/
FALCON_FUNC  List_len( ::Falcon::VMachine *vm )
{
   ItemList *list = dyncast<ItemList *>( vm->self().asObject()->getSequence() );
   vm->retval( (int64) list->size() );
}

}
}

/* end of list_ext.cpp */

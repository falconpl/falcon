/*
   FALCON - The Falcon Programming Language.
   FILE: list.cpp

   Implementation of the RTL List Falcon class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Implementation of the RTL List Falcon class.
*/


#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemlist.h>
#include <falcon/citerator.h>
#include <falcon/vm.h>
#include "core_module.h"
#include <falcon/eng_messages.h>

/*#
   
*/

namespace Falcon {
namespace core {

/*#
   @class List
   @brief Fast growable double linked list.
   @param ... An arbitrary list of parameters.

   The list class implements a double linked list of generic Falcon items that
   has some hooking with the falcon VM. Particularly, instances of the List
   class can be used as parameters for the Iterator constructor, or an iterator
   can be generated for them using first() and last() BOM methods. Also,
   instances of the List class can be used as any other sequence in for/in loops.

   In example, the following code:
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

/*#
   @init List
   This constructor creates a list that may be initially
   filled with the items passed as parameters. The items
   are inserted in the order they are passed.
*/
FALCON_FUNC  List_init ( ::Falcon::VMachine *vm )
{
   ItemList *list = new ItemList;
   int32 pc = vm->paramCount();
   for( int32 p = 0; p < pc; p++ )
   {
      list->push_back( *vm->param(p) );
   }

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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") ) );
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
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).
         extra("X") ) );
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
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->front() );
   list->pop_front();
}

/*#
   @method front List
   @brief Returns the first item in the list.
   @raise AccessError if the list is empty.
   @return The first item in the list.

   This method overloads the BOM method @a BOM.front. If the list
   is not empty, it returns the first element.
*/
FALCON_FUNC  List_front ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 ) // empty() is virtual
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->front() );
}

/*#
   @method back List
   @brief Returns the last item in the list.
   @raise AccessError if the list is empty.
   @return The last item in the list.

   This method overloads the BOM method @a BOM.back. If the list
   is not empty, it returns the last element.
*/
FALCON_FUNC  List_back ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  // empty() is virtual
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->back() );
}

/*#
   @method first List
   @brief Returns an iterator to the first element of the list.
   @return An iterator.

   Returns an iterator to the first element of the list. If the
   list is empty, an invalid iterator will be returned, but an
   insertion on that iterator will succeed and append an item to the list.
*/
FALCON_FUNC  List_first ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreObject *iobj = i_iclass->asClass()->createInstance();
   ItemListElement *iter = list->first();
   iobj->setUserData( new ItemListIterator( list, iter ) );
   iobj->setProperty( "_origin", vm->self() );
   vm->retval( iobj );
}

/*#
   @method last List
   @brief Returns an iterator to the last element of the list.
   @return An iterator.

   Returns an iterator to the last element of the list. If the
   list is empty, an invalid iterator will be returned, but an
   insertion on that iterator will succeed and append an item to the list.
*/
FALCON_FUNC  List_last ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreObject *iobj = i_iclass->asClass()->createInstance();
   iobj->setProperty( "_origin", vm->self() );

   ItemListElement *iter = list->last();
   iobj->setUserData( new ItemListIterator( list, iter ) );
   vm->retval( iobj );
}

/*#
   @method len List
   @brief Returns the number of items stored in the list.
   @return Count of items in the list.
*/
FALCON_FUNC  List_len( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   vm->retval( (int64) list->size() );
}

/*#
   @method empty List
   @brief Checks if the list is empty or not.
   @return True if the list is empty, false if contains some elements.
*/
FALCON_FUNC  List_empty( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   vm->retval( list->size() == 0 ); // empty is virtual
}

/*#
   @method clear List
   @brief Removes all the items from the list.
*/
FALCON_FUNC  List_clear( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   list->clear();
}

/*#
   @method erase List
   @brief Removes the item pointed by an iterator from the list.
   @param iter Iterator pointing at the element that must be removed.
   @raise AccessError if the iterator is invalid.

   If the given iterator is a valid iterator to an item of
   this class, the item is removed and the function returns true;
   the iterator is also moved forward as if its next() method was
   called. Other iterators currently active on this list are still
   valid after this call, except for those pointing to the deleted
   item which are invalidated.
*/
FALCON_FUNC  List_erase ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   Item *i_iter = vm->param(0);

   if ( i_iter == 0 || ! i_iter->isOfClass( "Iterator" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_iter ) ) ) );
      return;
   }

   CoreObject *iobj = i_iter->asObject();
   CoreIterator *iter = (CoreIterator *) iobj->getUserData();

   if ( ! list->erase( iter ) )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_iter ) ) ) );
   }
}

/*#
   @method insert List
   @brief Insert an item in a position specified by an iterator.
   @param it Iterator pointing where insertion will occur.
   @param item The item to be inserted.
   @raise AccessError if the iterator is invalid.

   Inserts an item at the position indicated by the iterator. After a successful
   insert, the new item is placed before the old one and the iterator points to the
   new item. In example, if inserting at the position indicated by first(), the new
   item is appended in front of the list and the iterator still points to the first
   item. To insert at the end of the list, get last() iterator, and move it forward
   with next(). The iterator is then invalidated, but an insert() will append an
   item at the end of the list, and the iterator will point to that item. To repeat
   insertions at the end, call next() after each insert.

   If the iterator is invalid (and not pointing to the last-past-one element), an
   access exception will be rasied.
*/

FALCON_FUNC  List_insert ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   Item *i_iter = vm->param(0);
   Item *i_item = vm->param(1);

   if ( i_iter == 0 || i_item == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( "O,X" ) ) );
      return;
   }

   if ( ! i_iter->isOfClass( "Iterator" ) )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_iter ) ) ) );
      return;
   }

   CoreObject *iobj = i_iter->asObject();
   CoreIterator *iter = (CoreIterator *) iobj->getUserData();

   // is the iterator a valid iterator on our item?
   if ( ! list->insert( iter, *i_item ) )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( rtl_invalid_iter ) ) ) );
   }
}

}
}

/* end of list.cpp */

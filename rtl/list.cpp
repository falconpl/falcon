/*
   FALCON - The Falcon Programming Language.
   FILE: list.cpp

   Implementation of the RTL List Falcon class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2007-12-01
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Implementation of the RTL List Falcon class.
*/


#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/itemlist.h>
#include <falcon/citerator.h>
#include <falcon/vm.h>
#include "falcon_rtl_ext.h"
#include "rtl_messages.h"

namespace Falcon {
namespace Ext {

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


FALCON_FUNC  List_pop ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  //empty() is virtual
   {
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->back() );
   list->pop_back();
}


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


FALCON_FUNC  List_popFront ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  //empty() is virtual
   {
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->front() );
   list->pop_front();
}


FALCON_FUNC  List_front ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 ) // empty() is virtual
   {
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->front() );
}

FALCON_FUNC  List_back ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   if( list->size() == 0 )  // empty() is virtual
   {
      vm->raiseModError( new RangeError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( list->back() );
}


FALCON_FUNC  List_first ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   Item *i_iclass = vm->findGlobalItem( "Iterator" );
   if ( i_iclass == 0 || ! i_iclass->isClass() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_iterator_not_found ) ) ) );
      return;
   }

   CoreObject *iobj = i_iclass->asClass()->createInstance();
   ItemListElement *iter = list->first();
   iobj->setUserData( new ItemListIterator( list, iter ) );
   vm->retval( iobj );
}


FALCON_FUNC  List_last ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );

   Item *i_iclass = vm->findGlobalItem( "Iterator" );
   if ( i_iclass == 0 || ! i_iclass->isClass() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_iterator_not_found ) ) ) );
      return;
   }

   CoreObject *iobj = i_iclass->asClass()->createInstance();
   iobj->setProperty( "origin", vm->self() );

   ItemListElement *iter = list->last();
   iobj->setUserData( new ItemListIterator( list, iter ) );
   vm->retval( iobj );
}


FALCON_FUNC  List_size( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   vm->retval( (int64) list->size() );
}


FALCON_FUNC  List_empty( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   vm->retval( list->size() == 0 ); // empty is virtual
}

FALCON_FUNC  List_clear( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   list->clear();
}

FALCON_FUNC  List_erase ( ::Falcon::VMachine *vm )
{
   ItemList *list = static_cast<ItemList *>( vm->self().asObject()->getUserData() );
   Item *i_iter = vm->param(0);

   if ( i_iter == 0 || ! i_iter->isOfClass( "Iterator" ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_iter ) ) ) );
      return;
   }

   CoreObject *iobj = i_iter->asObject();
   CoreIterator *iter = (CoreIterator *) iobj->getUserData();

   if ( ! list->erase( iter ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_iter ) ) ) );
   }
}


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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_iter ) ) ) );
      return;
   }

   CoreObject *iobj = i_iter->asObject();
   CoreIterator *iter = (CoreIterator *) iobj->getUserData();

   // is the iterator a valid iterator on our item?
   if ( ! list->insert( iter, *i_item ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
         origin( e_orig_runtime ).extra( vm->moduleString( msg::rtl_invalid_iter ) ) ) );
   }
}

}
}

/* end of list.cpp */

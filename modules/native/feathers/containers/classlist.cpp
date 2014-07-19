/* FALCON - The Falcon Programming Language.
 * FILE: classlist.cpp
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Sat, 01 Feb 2014 12:56:12 +0100
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2014: The above AUTHOR
 *
 * See LICENSE file for licensing details.
 */

#define SRC "modules/native/feathers/containers/classlist.cpp"
#include "classlist.h"
#include "list_mod.h"
#include "errors.h"

#include <falcon/function.h>
#include <falcon/vmcontext.h>

/*#
 @beginmodule containers
*/

namespace Falcon {
using namespace Mod;

namespace {
//=======================================================================================
// Class List
//

/*#
 @class List
 @optparam ... Items to be initially stored in the list.
 @brief Doubly linked list.
 @prop poolSize Size of pre-allocated element slots in the list.

*/

/*#
 @method push List
 @brief Appends an element at the end of the list.
 @param item The item to be pushed.
 @optparam ... Other items to be pushed
 @return The list itself.

*/
FALCON_DECLARE_FUNCTION( push, "item:X,..." )
FALCON_DEFINE_FUNCTION_P( push )
{
   if( pCount == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   List* cnt = ctx->tself<List>();
   for( int32 i = 0; i < pCount; ++i )
   {
      const Item& item = *ctx->param(i);
      cnt->push(item);
   }

   ctx->returnFrame(ctx->self());
}

/*#
 @method unshift List
 @brief Appends an element in front of the list.
 @param item The item to be inserted.
 @optparam ... Other items to be inserted.
 @return The list itself.

*/
FALCON_DECLARE_FUNCTION( unshift, "item:X,..." )
FALCON_DEFINE_FUNCTION_P( unshift )
{
   if( pCount == 0 )
   {
      throw paramError(__LINE__, SRC);
   }

   List* cnt = ctx->tself<List>();
   for( int32 i = 0; i < pCount; ++i )
   {
      const Item& item = *ctx->param(i);
      cnt->unshift(item);
   }

   ctx->returnFrame(ctx->self());
}

/*#
 @method pop List
 @brief Removes an element from the end of the list, and returns it
 @return The removed element
 @raise ContainerError if the container is empty.

*/
FALCON_DECLARE_FUNCTION( pop, "" )
FALCON_DEFINE_FUNCTION_P1( pop )
{
   List* cnt = ctx->tself<List>();
   Item last;
   if( ! cnt->back(last) )
   {
      throw FALCON_SIGN_ERROR(ContainerError, FALCON_ERROR_CONTAINERS_EMPTY );
   }

   cnt->pop();
   ctx->returnFrame(last);
}

/*#
 @method shift List
 @brief Removes an element from the beginning of the list, and returns it.
 @return The removed element.
 @raise ContainerError if the container is empty.

*/
FALCON_DECLARE_FUNCTION( shift, "" )
FALCON_DEFINE_FUNCTION_P1( shift )
{
   List* cnt = ctx->tself<List>();
   Item last;
   if( ! cnt->front(last) )
   {
      throw FALCON_SIGN_ERROR(ContainerError, FALCON_ERROR_CONTAINERS_EMPTY );
   }

   cnt->shift();
   ctx->returnFrame(last);
}


/*#
 @method front List
 @brief Removes an element from the beginning of the list, and returns it.
 @return The removed element.
 @raise ContainerError if the container is empty.

*/
FALCON_DECLARE_FUNCTION( front, "" )
FALCON_DEFINE_FUNCTION_P1( front )
{
   List* cnt = ctx->tself<List>();
   Item last;
   if( cnt->front(last) )
   {
      throw FALCON_SIGN_ERROR(ContainerError, FALCON_ERROR_CONTAINERS_EMPTY );
   }

   ctx->returnFrame(last);
}

/*#
 @method back List
 @brief Removes an element from the beginning of the list, and returns it.
 @return The removed element.
 @raise ContainerError if the container is empty.

*/
FALCON_DECLARE_FUNCTION( back, "" )
FALCON_DEFINE_FUNCTION_P1( back )
{
   List* cnt = ctx->tself<List>();
   Item last;
   if( cnt->back(last) )
   {
      throw FALCON_SIGN_ERROR(ContainerError, FALCON_ERROR_CONTAINERS_EMPTY );
   }

   ctx->returnFrame(last);
}


static void get_poolSize(const Class*, const String&, void* inst, Item& value )
{
   List* cnt = static_cast<List*>(inst);
   value.setInteger(cnt->poolSize());
}

static void set_poolSize(const Class*, const String&, void* inst, const Item& value )
{
   List* cnt = static_cast<List*>(inst);
   if( ! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR(AccessTypeError, e_inv_prop_value, .extra("N") );
   }

   cnt->poolSize((uint32) value.forceInteger());
}

}


ClassList::ClassList( const Class* parent ):
         ClassContainerBase("List")
{
   setParent(parent);
   addMethod( new FALCON_FUNCTION_NAME(push) );
   addMethod( new FALCON_FUNCTION_NAME(pop) );
   addMethod( new FALCON_FUNCTION_NAME(shift) );
   addMethod( new FALCON_FUNCTION_NAME(unshift) );
   addMethod( new FALCON_FUNCTION_NAME(front) );
   addMethod( new FALCON_FUNCTION_NAME(back) );

   addProperty( "poolSize", &get_poolSize, &set_poolSize );
}

ClassList::~ClassList()
{}

void* ClassList::createInstance() const
{
  return new List(this);
}

void ClassList::restore( VMContext* ctx, DataReader* ) const
{
   List* l = new List(this);
   ctx->pushData( Item(this, l) );
}




bool ClassList::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   List* list = static_cast<List*>(instance);
   if( pcount > 0 )
   {
      Item* items = ctx->opcodeParams(pcount);
      for ( int32 i = 0; i < pcount; ++i )
      {
         list->push(items[i]);
      }
   }

   return false;
}

}

/* end of classlist.cpp */


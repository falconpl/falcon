/*
   FALCON - The Falcon Programming Language.
   FILE: table.cpp

   Table support Iterface for Falcon.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Sep 2008 15:55:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/setup.h>
#include <falcon/coretable.h>
#include <falcon/vm.h>
#include "core_messages.h"

namespace Falcon {

/*#
   @class Table
   @brief Home of tabular programming.
*/

FALCON_FUNC Table_init( VMachine* vm )
{
   // the first parameter is the heading
   Item *i_heading = vm->param( 0 );
   if ( i_heading == 0 || ! i_heading->isArray() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR(rtl_no_tabhead) ) ) );
      return;
   }

   CoreTable* table = new CoreTable();

   if (! table->setHeader( i_heading->asArray() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR(rtl_invalid_tabhead) ) ) );
      return;
   }

   uint32 order = table->order();
   if ( order == 0 )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR(rtl_invalid_order) ) ) );
      return;
   }

   // create the first table
   table->insertPage( new CoreArray( vm, vm->paramCount() ) );
   table->setCurrentPage(0);

   // now we can safely add every other row that has been passed.
   for (int i = 1; i < vm->paramCount(); i ++ )
   {
      Item *vi = vm->param(i);
      if( ! vi->isArray() || vi->asArray()->length() != order )
      {
         String tempString = FAL_STR(rtl_invalid_tabrow);
         tempString += ": [";
         tempString.writeNumber( (int64)( i-1 ) );
         tempString += "]";
         vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
            .origin( e_orig_runtime )
            .extra( tempString ) ) );
         return;
      }

      table->insertRow( vi->asArray() );
   }

   vm->self().asObject()->setUserData( table );
}

/*#
   @method getHeader Table
   @brief Gets the name of one header, or the list of header namese.
   @optparam id If given, a number indicating the column of which to get the name.
   @return A string (if @id is given) or the vector of ordered column names.
*/
FALCON_FUNC Table_getHeader( VMachine* vm )
{
   CoreObject* self = vm->self().asObject();
   CoreTable* table = reinterpret_cast<CoreTable*>(self->getUserData());

   Item *i_pos = vm->param( 0 );
   if ( i_pos != 0 && ! i_pos->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "(N)" ) ) );
      return;
   }

   if( i_pos != 0 )
   {
      uint32 pos = (uint32) i_pos->forceInteger();
      if( pos > table->order() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ) ) );
         return;
      }

      vm->retval( new GarbageString( vm, table->heading(pos) ) );
   }
   else {
      CoreArray *ret = new CoreArray(vm, table->order() );

      for( uint32 i = 0; i < table->order(); i ++ )
      {
         ret->append( new GarbageString( vm, table->heading( i ) ) );
      }

      vm->retval( ret );
   }
}


/*#
   @method getColData Table
   @brief Gets the clumn wide data associated with this table.
   @optparam id If given, a number indicating the column of which to get the data.
   @return An item (if @id is given) or the vector of ordered column names.

*/
FALCON_FUNC Table_getColData( VMachine* vm )
{
   CoreObject* self = vm->self().asObject();
   CoreTable* table = reinterpret_cast<CoreTable*>(self->getUserData());

   Item *i_pos = vm->param( 0 );
   if ( i_pos != 0 && ! i_pos->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "(N)" ) ) );
      return;
   }

   if( i_pos != 0 )
   {
      uint32 pos = (uint32) i_pos->forceInteger();
      if( pos > table->order() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ) ) );
         return;
      }

      vm->retval( table->columnData(pos) );
   }
   else {
      CoreArray *ret = new CoreArray(vm, table->order() );

      for( uint32 i = 0; i < table->order(); i ++ )
      {
         ret->append( table->columnData(i) );
      }

      vm->retval( ret );
   }
}


FALCON_FUNC Table_order( VMachine* vm )
{
   CoreObject* self = vm->self().asObject();
   CoreTable* table = reinterpret_cast<CoreTable*>(self->getUserData());
   vm->retval( (int64) table->order() );
}

}

/* end of table.cpp */

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
   CoreArray *page = new CoreArray( vm, vm->paramCount() );
   table->insertPage( page );
   table->setCurrentPage(0);
   CoreObject *self = vm->self().asObject();

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
      vi->asArray()->table( self );

      table->insertRow( vi->asArray() );
   }

   self->setUserData( table );
}

/*#
   @method getHeader Table
   @brief Gets the name of one header, or the list of header names.
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

/*#
   @method order Table
   @brief Returns the order of the table (column count).
   @return The number of the columns, and of the length of every array in the table.
*/
FALCON_FUNC Table_order( VMachine* vm )
{
   CoreObject* self = vm->self().asObject();
   CoreTable* table = reinterpret_cast<CoreTable*>(self->getUserData());
   vm->retval( (int64) table->order() );
}

/*#
   @method len Table
   @brief Returns the length of the table (the number of rows).
   @return The rows in the current page of the table.

   Tables may have multiple pages, each of which having the same order
   (column count), but different length (rows).

   This method returns the length of the currently active page.
*/
FALCON_FUNC Table_len( VMachine* vm )
{
   CoreObject* self = vm->self().asObject();
   CoreTable* table = reinterpret_cast<CoreTable*>(self->getUserData());
   if ( table->currentPage() == 0 ) {
      vm->retval( 0 );
   }
   else {
      vm->retval( (int64) table->currentPage()->length() );
   }
}


/*#
   @method front Table
   @brief Returns the first item in the table.
   @raise AccessError if the table is empty.
   @return The first item in the table.

   This method overloads the BOM method @a BOM.front. If the table
   is not empty, it returns the first element.
*/
FALCON_FUNC  Table_front ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );

   if( table->empty() ) // empty() is virtual
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( table->front() );
}

/*#
   @method back Table
   @brief Returns the last item in the table.
   @raise AccessError if the table is empty.
   @return The last item in the table.

   This method overloads the BOM method @a BOM.back. If the table
   is not empty, it returns the last element.
*/
FALCON_FUNC  Table_back ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );

   if( table->empty() )  // empty() is virtual
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }

   vm->retval( table->back() );
}

static void internal_first_last( VMachine *vm, bool mode )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );

   if( table->empty() )  // empty() is virtual
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ ).
         origin( e_orig_runtime ) ) );
      return;
   }


   Item *i_iclass = vm->findWKI( "Iterator" );
   fassert( i_iclass != 0 );

   CoreObject *iobj = i_iclass->asClass()->createInstance();
   iobj->setUserData( table->getIterator(mode) );
   vm->retval( iobj );
}

/*#
   @method first Table
   @brief Returns an iterator to the first element of the table.
   @return An iterator.

   Returns an iterator to the first element of the table. If the
   table is empty, an invalid iterator will be returned, but an
   insertion on that iterator will succeed and append an item to the table.
*/
FALCON_FUNC  Table_first ( ::Falcon::VMachine *vm )
{
   internal_first_last( vm, false );
}

/*#
   @method last Table
   @brief Returns an iterator to the last element of the table.
   @return An iterator.

   Returns an iterator to the last element of the table. If the
   table is empty, an invalid iterator will be returned, but an
   insertion on that iterator will succeed and append an item to the table.
*/
FALCON_FUNC  Table_last ( ::Falcon::VMachine *vm )
{
   internal_first_last( vm, true );
}

static uint32 internal_col_pos( CoreTable *table, VMachine *vm, Item *i_column )
{
   uint32 colPos;

   if( i_column->isString() ) {
      colPos = table->getHeaderPos( *i_column->asString() );
      if ( colPos == CoreTable::noitem )
      {
         // there isn't such field
         vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ )
            .origin( e_orig_runtime )
            .extra( *i_column->asString() ) ) );
         return CoreTable::noitem;
      }
   }
   else {
      colPos = (uint32) i_column->forceInteger();
      if ( colPos >= table->order() )
      {
         String temp;
         temp = "col ";
         temp.append( colPos );
         vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ )
            .origin( e_orig_runtime )
            .extra( temp ) ) );
         return CoreTable::noitem;
      }
   }

   return colPos;
}

static void internal_get_item( CoreTable *table, CoreArray *row, VMachine *vm, Item *i_column )
{
   uint32 colPos = internal_col_pos( table, vm, i_column );

   // otherwise we have already an error risen.
   if ( colPos != CoreTable::noitem )
   {
      Item ret = (*row)[colPos];

      if ( ret.isNil() && ! ret.isOob() )
         ret = *table->getHeaderData(colPos);

      // eventually methodize.
      if ( ret.isFunction() )
         ret.setTabMethod( row, ret.asFunction(), ret.asModule() );

      vm->retval( ret );
   }
}


/*#
   @method get Table
   @brief Gets a row in a table.
   @param row a Row number.
   @optparam tcol The name of the column to be extracted (target column; either name or 0 based number).
   @return An array (if the column is not specified) or an item.

   The returned array is a "table component", and as such, its size cannot be changed;
   also, it inherits all the table clumns, that can be accessed as bindings with the
   dot accessor and will resolve in one of the element in the array.
*/
FALCON_FUNC  Table_get ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_pos = vm->param(0);
   Item* i_column = vm->param(1);

   if ( i_pos == 0 || ! i_pos->isOrdinal()
      || ( i_column != 0 && ! (i_column->isString() || i_column->isOrdinal()) ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N, [S|N]" ) ) );
      return;
   }

   CoreArray* page = table->currentPage();

   // Get the correct row.
   uint32 pos = (uint32) i_pos->forceInteger();
   if ( pos >= page->length() )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ )
         .origin( e_orig_runtime ) ) );
      return;
   }

   // Should we also get a single item?
   if( i_column == 0 )
   {
      Item &itm = (*page)[pos];
      fassert( itm.isArray() );
      itm.asArray()->tablePos( pos );
      vm->retval( itm );
   }
   else {
      internal_get_item( table, (*page)[pos].asArray(), vm, i_column );
   }
}

/*#
   @method find Table
   @brief Finds an element in a table.
   @param column The column where to perform the search (either name or 0 based number).
   @param value The value to be found.
   @optparam tcol The name of the column to be extracted (target column; either name or 0 based number).
   @return An array (if the column is not specified) or an item.

   The returned array is a "table component", and as such, its size cannot be changed;
   also, it inherits all the table clumns, that can be accessed as bindings with the
   dot accessor and will resolve in one of the element in the array.

   In case of success, through the BOM method @a BOM.tabRow it is possible to retreive
   the table row position of the returned array.
*/
FALCON_FUNC  Table_find ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);
   Item* i_value = vm->param(1);
   Item* i_tcol = vm->param(2);

   if ( i_column == 0 || ! ( i_column->isString() || i_column->isOrdinal() ) ||
        i_value == 0 ||
        ( i_tcol != 0 && ! (i_tcol->isString()|| i_tcol->isOrdinal()) ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S|N,X,[S|N]" ) ) );
      return;
   }

   CoreArray* page = table->currentPage();
   uint32 pos = CoreTable::noitem;

   // the easy way
   // must be a future binding. find it.
   uint32 col = internal_col_pos( table, vm, i_column );
   if ( col == CoreTable::noitem )
   {
      // error already risen.
      return;
   }

   for ( uint32 i = 0; i < page->length(); i++ )
   {
      if( vm->compareItems( page->at(i).asArray()->at(col), *i_value ) == 0 )
      {
         pos = i;
         break;
      }

      if ( vm->hadError() )
         return;
   }

   if ( pos == CoreTable::noitem )
   {
      // there isn't such field
      vm->raiseModError( new AccessError( ErrorParam( e_prop_acc, __LINE__ )
         .origin( e_orig_runtime )));
      return;
   }

   // we know we have a valid pos here.
   if( i_tcol == 0 )
   {
      Item &itm = (*page)[pos];
      fassert( itm.isArray() );
      itm.asArray()->tablePos( pos );
      vm->retval( itm );
   }
   else {
      internal_get_item( table, (*page)[pos].asArray(), vm, i_tcol );
   }
}

/*#
   @method insert Table
   @brief Insert a row in the table.
   @param element The row to be inserted.
   @param pos The position where to insert the row.
   @raise AccessError if the position is out of range.
   @raise ParamError if the row is not an array with the same lenght of the table order.

   The element is inserted before the given position.

   If @b pos is greater or equal to the length of the table, the row will be inserted
   at end (added). If @b pos is negative, the row will be accessed backward, -1 being
   the last element (the row will be inserted before the last one).
*/
FALCON_FUNC  Table_insert ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_element = vm->param(0);
   Item* i_pos = vm->param(1);

   if ( i_element == 0 || ! i_element->isArray()
        || i_pos == 0 || ! i_pos->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "A,N" ) ) );
      return;
   }

   CoreArray* element = i_element->asArray();
   uint32 pos = i_pos->forceInteger();
   if ( element->length() != table->order() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_invalid_tabrow ) ) ) );
      return;
   }

   CoreArray* page = table->currentPage();
   if ( pos < 0 )
      pos = page->length() - pos;

   if ( pos > page->length() ) {
      page->append( element );
   }
   else {
      page->insert( Item(element), pos );
   }

   element->table( vm->self().asObject() );
}

}

/* end of table.cpp */

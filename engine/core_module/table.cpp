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
   table->insertPage( vm->self().asObject(), page );
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
   @return A string (if @b id is given) or the vector of ordered column names.
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
      ret->resize( table->order() );

      for( uint32 i = 0; i < table->order(); i ++ )
      {
         ret->at(i) = new GarbageString( vm, table->heading( i ) );
      }

      vm->retval( ret );
   }
}


/*#
   @method getColData Table
   @brief Gets the clumn wide data associated with this table.
   @optparam id If given, a number indicating the column of which to get the data.
   @return An item (if @b id is given) or the vector of ordered column names.

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
         temp.writeNumber( (int64) colPos );
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
   if ( i_column->isInteger() && (((uint32)i_column->asInteger()) == CoreTable::noitem) )
   {
      vm->retval( row );
      return;
   }

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
   @method columnPos Table
   @brief Returns the number of a given column name.
   @param column The column header name.
   @return The numeric position of the column or -1 if not found.
*/
FALCON_FUNC  Table_columnPos ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);

   if ( i_column != 0 && ! i_column->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) ) );
      return;
   }

   uint32 colpos = table->getHeaderPos( *i_column->asString() );
   if ( colpos == CoreTable::noitem )
      vm->regA().setInteger( -1 );
   else
      vm->regA().setInteger( colpos );
}

/*#
   @method columnData Table
   @brief Returns the column data bound with a certain column
   @param column The column header name or numeric position.
   @return The column data for the given column, or nil is not found.

   Notice that the column data of an existing column may be nil; to know
   if a column with a given name exists, use the @a Table.column method.
*/

FALCON_FUNC  Table_columnData ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);

   if ( i_column != 0 && ! i_column->isString() && ! i_column->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "S" ) ) );
      return;
   }

   uint32 colpos = internal_col_pos( table, vm, i_column );
   if ( colpos == CoreTable::noitem )
      vm->regA().setNil();
   else
      vm->regA() = table->columnData( colpos );
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
   @param row The position where to insert the row.
   @param element The row to be inserted.
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
   Item* i_pos = vm->param(0);
   Item* i_element = vm->param(1);

   if (
        i_pos == 0 || ! ( i_pos->isOrdinal() || i_pos->isNil() )
        || i_element == 0 || ! i_element->isArray()
        )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N,A" ) ) );
      return;
   }

   CoreArray* element = i_element->asArray();
   uint32 pos = (uint32) i_pos->isNil() ? table->order() : i_pos->forceInteger();
   if ( element->length() != table->order() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_invalid_tabrow ) ) ) );
      return;
   }

   CoreArray* page = table->currentPage();
   if ( pos > table->order() )
      pos = table->order();

   page->insert( Item(element), pos );

   element->table( vm->self().asObject() );
}

/*#
   @method remove Table
   @brief Remove a row from the table.
   @param row The number of the row to be removed.
   @raise AccessError if the position is out of range.
   @return The removed array.

   This method removes one of the rows from the table.

   However, the array still remember the table from which it came from,
   as the table may have multiple reference to the array. So, the array
   stays bound with the table, and cannot be modified.
*/
FALCON_FUNC  Table_remove ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_row = vm->param(0);

   if (i_row == 0 || ! i_row->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N" ) ) );
      return;
   }

   uint32 pos = (uint32) i_row->forceInteger();

   CoreArray* page = table->currentPage();
   if ( pos < 0 )
      pos = page->length() - pos;

   if ( pos >= page->length() ) {
      vm->raiseModError( new AccessError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime ) ) );
      return;
   }

   CoreArray *rem = (*page)[pos].asArray();
   //rem->table(0);
   page->remove( pos );
   vm->retval(rem);
}


/*#
   @method setColumn Table
   @brief Change the title or column data of a column.
   @param column The number of name of the column to be renamed.
   @param name The new name of the column, or nil to let it unchanged.
   @optparam coldata The new column data.

   This method changes the column heading in a table. It may be also used
   to change the column-wide data.
*/
FALCON_FUNC  Table_setColumn ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);
   Item* i_name = vm->param(1);
   Item* i_value = vm->param(2);

   if (i_column == 0 || ! ( i_column->isOrdinal()|| i_column->isString())
      || i_name == 0 || ! ( i_name->isNil() || i_name->isString()) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N|S, Nil|S, [X]" ) ) );
      return;
   }

   uint32 colpos = internal_col_pos( table, vm, i_column );
   if ( colpos == CoreTable::noitem )
   {
      return;
   }

   if ( ! i_name->isNil() )
   {
      table->renameColumn( colpos, *i_name->asString() );
   }

   if ( i_value != 0 )
   {
      table->columnData( colpos ) = *i_value;
   }
}

/*#
   @method insertColumn Table
   @brief Inserts a column in a table.
   @param column The column name or position where to insert the column.
   @param name The name of the new column.
   @optparam coldata The column data for the new column.
   @optparam dflt Default value for the newly inserted columns.

   This method creates a new column in the table, inserting or adding
   a new heading, a new column data and an item in the coresponding
   column position of each array in the table.

   If @b dflt parameter is specified, that value is used to fill the
   newly created columns in the table rows, otherwise the new items
   will be nil.
*/
FALCON_FUNC  Table_insertColumn ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);
   Item* i_name = vm->param(1);
   Item* i_data = vm->param(2);
   Item* i_dflt = vm->param(3);

   if ( i_column == 0 || ! ( i_column->isOrdinal()|| i_column->isString())
      || i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N|S, S, [X], [X]" ) ) );
      return;
   }

   uint32 colpos;
   if ( i_column->isOrdinal() )
   {
      colpos = i_column->forceInteger() < 0 ?
         (uint32) (table->order() + i_column->forceInteger()) :
         (uint32) (i_column->forceInteger());

      if ( colpos > table->order() )
         colpos = table->order();
   }
   else {
      colpos = internal_col_pos( table, vm, i_column );
      if ( colpos == CoreTable::noitem )
      {
         // already raised.
         return;
      }
   }

   Item data, dflt;
   if ( i_data != 0 )
      data = *i_data;
   if ( i_dflt != 0 )
      dflt = *i_dflt;

   table->insertColumn( colpos, *i_name->asString(), data, dflt );
}

/*#
   @method removeColumn Table
   @brief Inserts a column in a table.
   @param column The column name or position to be removed.
   @raise AccessError if the column is not found.

   This method removes column in the table, removing also the coresponding
   position in the

   If @b dflt parameter is specified, that value is used to fill the
   newly created columns in the table rows, otherwise the new items
   will be nil.
*/
FALCON_FUNC  Table_removeColumn ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_column = vm->param(0);


   if ( i_column == 0 || ! ( i_column->isOrdinal() || i_column->isString()) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N|S" ) ) );
      return;
   }

   if ( i_column->isOrdinal() && i_column->forceInteger() < 0 )
   {
      i_column->setInteger( table->order() + i_column->forceInteger());
   }

   uint32 colpos = internal_col_pos( table, vm, i_column );
   if ( colpos == CoreTable::noitem )
   {
      // already raised.
      return;
   }


   table->removeColumn( colpos );
}

static bool table_choice_next( Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   CoreArray &page = *table->currentPage();
   uint32 start = (uint32) vm->local(0)->forceInteger();
   uint32 row = (uint32) vm->local(1)->forceInteger();
   uint32 end = (uint32) vm->local(2)->forceInteger();

   // has the evaluation function just asked us to stop?
   if( vm->regA().isOob() )
   {
      // in case of just row end, A can't be nil as the function has
      if( vm->regA().isNil() )
      {
         vm->regA().setOob(false);
      }
      else {
         // get the winner
         CoreArray *winner = page[row-1].asArray();
         winner->tablePos( row-1 );
         internal_get_item( table, winner, vm, vm->local(3) );
      }

      // we're done, don't call us anymore
      return false;
   }
   else if ( start != row ) {
      // we have a bidding
      uint32 pos = row - start;

      // a bit paranoid, but users may really screw up the table.
      if ( pos > table->biddingsSize() ) {
         vm->raiseModError( new AccessError( ErrorParam( e_continue_out, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_broken_table ) ) ) );
         return false;
      }

      numeric *biddings = table->biddings();
      biddings[pos-1] = vm->regA().forceNumeric();

      // is the bidding step over?
      if( pos == end - start) {
         uint32 maxrow = CoreTable::noitem;

         // our threshold is 0. If no item meets the requirement, we return nil
         numeric maxval = 0.0;
         // but take also the first row in the loop, to update maxval
         for ( pos = 0; pos < end - start; pos ++ )
         {
            if( biddings[pos] > maxval )
            {
               maxrow = pos + start;
               maxval = biddings[pos];
            }
         }

         // no winner?
         if ( maxrow == CoreTable::noitem )
         {
            vm->retnil();
         }
         else {
            // we have a winner.
            CoreArray *winner = page[row-1].asArray();
            winner->tablePos( row-1 );
            internal_get_item( table, page[maxrow].asArray(), vm, vm->local(3) );
         }

         // we're done, don't call us anymore
         return false;
      }
   }

   // nothing special to do: just call the desired function with our row.
   Item &rowItem = page[row];
   rowItem.asArray()->tablePos(row); // save the position, in case is needed by our evaluator

   // update counter for next loop
   vm->local(1)->setInteger( row + 1 );

   // do we have to call a function or an item in the row?
   Item &calling = *vm->local(4);
   if( calling.isInteger() )
   {
      // colunn in the table.
      Item ret = page[row].asArray()->at( (uint32) calling.asInteger() );

      if ( ret.isNil() && ! ret.isOob() )
         ret = *table->getHeaderData((uint32) calling.asInteger());

      // eventually methodize.
      if ( ret.isFunction() )
      {
         ret.setTabMethod( page[row].asArray(), ret.asFunction(), ret.asModule() );
         vm->callFrame( ret, 0 );
         return true;
      }

      if ( ret.isCallable() )
      {
         vm->pushParameter( page[row] );
         vm->callFrame( ret, 1 );
         return true;
      }

      // else, the item is not callable! Raise an error.
      // (it's ok also if we pushed the parameter; stack is unwinded).
      vm->raiseModError( new AccessError( ErrorParam( e_non_callable, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_uncallable_col ) ) ) );

      // pitifully, we're done.
      return false;
   }
   else {
      // function provided externally
      vm->pushParameter( page[row] );
      vm->callFrame( calling, 1 );
   }

   // call us again when the frame is done.
   return true;
}

static void internal_bind_or_choice( VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   Item* i_offer = vm->param(1);
   Item* i_rows = vm->param(2);

   CoreArray *cp = table->currentPage();

   // if the table is empty, just return.
   if ( cp->length() == 0 )
   {
      vm->retnil();
      return;
   }

   uint32 start, end;
   if( i_rows == 0 )
   {
      start = 0;
      end = cp->length();
   }
   else {
      start = (uint32) (i_rows->asRangeStart() < 0 ?
            cp->length() + i_rows->asRangeStart() : i_rows->asRangeStart());
      end = i_rows->asRangeIsOpen() ? cp->length() :
          (uint32) (i_rows->asRangeEnd() < 0 ?
            cp->length() + i_rows->asRangeEnd() : i_rows->asRangeEnd());
   }


   if ( start == cp->length() || start >= end || start < 0 )
   {
      vm->retnil();
      return;
   }

   // performs also a preliminary check of column pos
   uint32 colpos;
   if( i_offer == 0 || i_offer->isNil() )
   {
      colpos = CoreTable::noitem;
   }
   else {
      colpos = internal_col_pos( table, vm, i_offer );
      if ( colpos == CoreTable::noitem )
      {
         // error already raised
         return;
      }
   }

   // locals already allocated.
   vm->local(0)->setInteger( (int64) start ); // we need a copy
   vm->local(1)->setInteger( (int64) start );
   vm->local(2)->setInteger( (int64) end );
   vm->local(3)->setInteger( (int64) colpos );

   table->reserveBiddings( end - start + 1);

   // returning from this frame will call table_choice_next
   vm->returnHandler( table_choice_next );
}

/*#
   @method choice Table
   @brief Performs a choice between rows.
   @param func Choice function or callable.
   @optparam offer Offer column (number or name).
   @optparam rows Range of rows in which to perform the bidding.
   @return The winning row, or the coresponding value in the offer column.

   This method sends all the rows in the table as the sole parameter of
   a function which has to return a numeric value for each row.

   After all the rows have been processed, the row for which the called function
   had the highest value is returned. If the optional parameter @b offer is specified,
   then the item specified by that column number or name is returned instead.

   If two or more rows are equally evaluated by the choice function, only the
   first one is returned.

   The evaluation function may return a number lower than 0 to have the row effectively
   excluded from the evaluation. If all the rows are evaluated to have a value lower
   than zero, the function returns nil (as if the table was empty).

   The function may force an immediate selection by returning an out of band item.
   An out of band nil will force this method to return nil, and an out of band
   number will force the selection of the current row.

   If an @b offer parameter is specified, then the item in the coresponding
   column (indicated by a numeric index or by column name) is returned.

   A @b row range can be used to iterate selectively on one part of the table.

   If the table is empty, or if the given row range is empty, the function returns
   nil.

   @note Except for OOB nil, every return value coming from the choice function
   will be turned into a floating point numeric value. Non-numeric returns will
   be evaluated as 0.0.
*/
FALCON_FUNC  Table_choice ( ::Falcon::VMachine *vm )
{
   Item* i_func = vm->param(0);
   Item* i_offer = vm->param(1);
   Item* i_rows = vm->param(2);

   if ( i_func == 0 || ! i_func->isCallable()
         || (i_offer != 0 && (! i_offer->isNil() && ! i_offer->isString() && ! i_offer->isOrdinal() ))
         || (i_rows != 0 && ! i_rows->isRange() )
       )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "C,[N|S],[R]" ) ) );
      return;
   }

   // prepare local stack and function pointer
   vm->addLocals(5);
   *vm->local(4) = *vm->param(0);

   internal_bind_or_choice( vm );
}

/*#
   @method bidding Table
   @brief Performs a bidding between rows.
   @param column Betting column (number or name).
   @optparam offer Offer column (number or name).
   @optparam rows Range of rows in which to perform the bidding.
   @return The winning row, or the coresponding value in the offer column.
   @return AccessError if the table or ranges are empty.

   This method calls iteratively all the items in a determined column of
   the table, recording their return value, which must be numeric. It is
   allowed also to have numeric and nil values in the cells in the
   betting column. Numeric values will be considered as final values and
   will participate in the final auction, while row having nil values
   will be excluded.

   After the bidding is complete, the row offering the highest value
   is selected and returned, or if the @b offer parameter is specified,
   the coresponding value in the given column will be returned.

   @note If the bidding element is a plain function, it will be called
      as a method of the array, so "self" will be available. In all the
      other cases, the array will be passed as the last parameter.

   A @b rows range can be specified to limit the bidding to a part
   smaller part of the table.
*/
FALCON_FUNC  Table_bidding ( ::Falcon::VMachine *vm )
{
   Item* i_column = vm->param(0);
   Item* i_offer = vm->param(1);
   Item* i_rows = vm->param(2);

   if ( i_column == 0 || ( ! i_column->isOrdinal() && ! i_column->isString() )
         || (i_offer != 0 && (! i_offer->isNil() && ! i_offer->isString() && ! i_offer->isOrdinal() ))
         || (i_rows != 0 && ! i_rows->isRange() )
       )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "[N|S],[N|S],[R]" ) ) );
      return;
   }

   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   uint32 colpos = internal_col_pos( table, vm, i_column );
   // wrong position?
   if( colpos == CoreTable::noitem )
      return;

   // prepare local stack and data pointer
   vm->addLocals(5);
   vm->local(4)->setInteger( colpos );

   internal_bind_or_choice( vm );
}

/*#
   @method pageCount Table
   @brief Gets the number of pages in this table.
   @return Number of pages stored in this table.
*/
FALCON_FUNC  Table_pageCount ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   vm->retval( (int64) table->pageCount() );
}

/*#
   @method setPage Table
   @brief Sets current active page.
   @param pageId The number of the selected page.

   All the tables are created with at least one page having ID = 0.
*/
FALCON_FUNC  Table_setPage ( ::Falcon::VMachine *vm )
{
   Item* i_page = vm->param(0);

   if ( i_page == 0 || ! i_page->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N" ) ) );
      return;
   }

   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   uint32 pcount = table->pageCount();
   int64 reqPid = i_page->forceInteger();
   uint32 pid = (uint32)(reqPid < 0 ? pcount + reqPid : reqPid);

   if ( ! table->setCurrentPage( pid ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_param_range, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_no_page ) ) ) );
      return;
   }
}

/*#
   @method curPage Table
   @brief Gets the current table page.
   @return The currently active table page.

   All the tables are created with at least one page having ID = 0.
*/
FALCON_FUNC  Table_curPage ( ::Falcon::VMachine *vm )
{
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   vm->retval( (int64) table->currentPageId() );
}

/*#
   @method insertPage Table
   @brief Inserts a new table page.
   @optparam pageId The position at which to insert the page.
   @optparam data an array of rows (arrays), each of which having length equal to table order.

   If @b pos is greater than the number of pages in the table, or not given, the page will be
   appended at the end.
*/
FALCON_FUNC  Table_insertPage ( ::Falcon::VMachine *vm )
{
   Item* i_pos = vm->param(0);
   Item* i_data = vm->param(1);

   if ( i_pos != 0 && ! ( i_pos->isOrdinal() || i_pos->isNil() )
      || ( i_data != 0 && ! i_data->isArray() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "[N],[A]" ) ) );
      return;
   }

   uint32 pos = i_pos == 0 || i_pos->isNil() ? CoreTable::noitem : i_pos->forceInteger();
   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   if ( i_data == 0 )
   {
      table->insertPage( vm->self().asObject(), new CoreArray(vm), pos );
   }
   else {
      if ( ! table->insertPage( vm->self().asObject(), i_data->asArray()->clone(), pos ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_param_type, __LINE__ )
            .origin( e_orig_runtime )
            .extra( FAL_STR( rtl_invalid_tabrow ) ) ));
         return;
      }
   }
}

/*#
   @method removePage Table
   @brief Removes a page.
   @param pageId The page to be removed.

   The table cannot exist without at least one page, and if the
   deleted page is the current one, the page 0 is selected.
*/
FALCON_FUNC  Table_removePage ( ::Falcon::VMachine *vm )
{
   Item* i_pos = vm->param(0);

   if ( i_pos == 0 || ! i_pos->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .origin( e_orig_runtime )
         .extra( "N" ) ) );
      return;
   }

   CoreTable *table = static_cast<CoreTable *>( vm->self().asObject()->getUserData() );
   if ( ! table->removePage( (uint32) i_pos->forceInteger() ) )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_arracc, __LINE__ )
         .origin( e_orig_runtime )
         .extra( FAL_STR( rtl_no_page ) ) ) );
   }
}

}

/* end of table.cpp */

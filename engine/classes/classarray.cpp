/*
   FALCON - The Falcon Programming Language.
   FILE: classarray.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classarray.cpp"

#include <falcon/classes/classarray.h>
#include <falcon/range.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/itemarray.h>
#include <falcon/datareader.h>
#include <falcon/datawriter.h>
#include <falcon/stdhandlers.h>
#include <falcon/function.h>

#include <falcon/stderrors.h>

namespace Falcon {

static void get_len( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<ItemArray*>( instance )->length();
}

static void get_allocated( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<ItemArray*>( instance )->allocated();
}

static void get_empty( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean(static_cast<ItemArray*>( instance )->length() == 0);
}

static void get_growth( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<ItemArray*>( instance )->growth();
}

static void set_growth( const Class*, const String&, void* instance, const Item& value )
{
   if(! value.isOrdinal() )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_inv_params, .extra("N"));
   }

   int64 growth =  value.forceInteger();
   if( growth < 0 )
   {
      growth = 0;
   }

   static_cast<ItemArray*>( instance )->growth((length_t)growth);
}

namespace _classArray {

/**
 @class Array
 @brief Class of Falcon language array []
 @param ... The elements to be initially stored in the array.

 @prop len Length of the array
 @prop allocated Number of items that can be stored in this array before new memory is required.
 @prop empty true if there isn't any element in the array
*/
 /*
    @property growth Array
    @brief Allocation growth size control.

    This property is used when there is the need to allocate new memory
    to store items inserted in the array.

    Growth allocation can be discrete (if this property is greater than zero)
    or exponential (if this property is zero).

    If @b growth is nonzero, each time new memory is needed, a number of
    items multiple of @b growth greater or equal to the required size
    are allocated. For instance, if @b growth is 32, and 50 items must be
    stored, the array pre-allocates 64 items. As the 65th item is added,
    96 items are pre-allocated. If 1025 items are to be stored, 1024+32=1056
    items are pre-allocated.

    If @b growth is zero, when new space is needed, Falcon allocates the nearest
    power of 2 in excess that could hold the required data. For instance, if the
    array needs to store 65 items, then 128 items are preallocated. If it needs
    to store 1025 items, 2048 items are preallocated.

    The first method is more suitable when allocations are performed in small
    increments, and the size of the array rarely changes. The second method is
    more efficient when inserting a great number of items one at a time, as it
    requires less reallocations, but it can eat up a lot of memory.

    @note, the @a Array.compact method can be used to cut down the memory to the
    required size once inserted a great number of items.

    When creating a big array by inserting one item at a time, the ideal policy would be
    that of setting growth to zero, inserting the needed item, then setting growth
    back to a reasonable size, as 16 or 32, and calling  @a Array.compact to
    reduce memory usage.

    @note By default @b growth is set to 16 on newly created arrays.
 */
FALCON_DECLARE_FUNCTION(init,"...");
void Function_init::invoke(VMContext* ctx, int32 pCount)
{

   ItemArray* array = new ItemArray;
   if( pCount > 0 )
   {
      array->reserve(pCount);

      for(int32 p = 0; p < pCount; ++p ) {
         array->append(*ctx->param(p));
      }
   }

   ctx->returnFrame(FALCON_GC_HANDLE(array));
}


static void internal_fill( ItemArray* array, Item* i_item, length_t start, length_t count )
{
   if( ! i_item->isUser() || (i_item->isString() && i_item->asString()->isImmutable()) )
   {
      // is flat, do a flat copy
      for( length_t pos = start; pos < start+count; ++pos )
      {
         (*array)[pos] = *i_item;
      }
   }
   else
   {
      Class* cls = 0;
      void* data = 0;
      i_item->asClassInst(cls, data);

      for( length_t pos = start; pos < start+count; ++pos )
      {
         void* clone = cls->clone(data);

         if( clone == 0 )
         {
            throw FALCON_SIGN_XERROR(ParamError, e_param_type,
                     .extra("Default item not cloneable"));
         }

         (*array)[pos] = FALCON_GC_STORE(cls, clone);
      }
   }
}

/**
 @method buffer Array
 @brief (static) Creates an array of a given size, optionally filled with a given element.
 @param size Initial size of the array.
 @optparam dflt Initializer
 @retrun A new array, eventually filled with copies of the given item.

 If the @b dflt item is not given, all the elements in the returned array are
 set to @b nil. If it's given, each element is a clone'd entity of the given item.

 */
FALCON_DECLARE_FUNCTION(buffer,"size:N>=0,dflt:[X]");
void Function_buffer::invoke(VMContext* ctx, int32 )
{
   Item* i_size = ctx->param(0);
   Item* i_dflt = ctx->param(1);

   int64 len;
   if( i_size == 0 || ! i_size->isOrdinal() || (len = i_size->asInteger()) < 0 )
   {
      throw paramError(__LINE__, SRC );
   }

   ItemArray* array = new ItemArray;
   // pass it to the GC now, because we might throw
   FALCON_GC_HANDLE(array);

   if( len > 0 )
   {
      array->resize(static_cast<length_t>(len));

      if( i_dflt != 0 )
      {
         internal_fill(array, i_dflt, 0, static_cast<length_t>(len) );
      }
   }

   ctx->returnFrame( Item(array->handler(), array) );
}


/**
 @method alloc Array
 @brief (static) Preallocates an empty array to a given size.
 @param size The size that must be preallocated.

 This static method of the Array class returns an empty
 array that is guaranteed not to be reallocated up to at least
 @b size items are added to it.

 It's equivalent to
 @code
 a = Array()
 a.reserve(size)
 @endcode

 */
FALCON_DECLARE_FUNCTION(alloc,"size:[N>=0]");
void Function_alloc::invoke(VMContext* ctx, int32 )
{
   Item* i_size = ctx->param(0);

   if( i_size != 0 && ! i_size->isOrdinal() )
   {
      throw paramError(__LINE__, SRC );
   }

   if( i_size == 0 )
   {
      ItemArray* array = new ItemArray;
      ctx->returnFrame(FALCON_GC_HANDLE(array));
   }
   else {
      if( i_size->forceInteger() < 0 )
      {
         throw paramError(__LINE__, SRC );
      }
      else
      {
         ItemArray* array = new ItemArray;
         array->reserve((length_t)i_size->forceInteger());
         ctx->returnFrame(FALCON_GC_HANDLE(array));
      }
   }
}

/**
 @method reserve Array
 @brief Allocates more memory to store new data without resizing the array.
 @param size The size that must be preallocated.
 @return The same array.

 The memory occupied by this array is enlarged to the required size
 so that the array that is guaranteed not to be reallocated up to at least
 @b size items are added to it.

 If the size is smaller than the current array size, nothing is done.

 @note When used statically, this method takes a target array as first parameter.

 It's equivalent to
 @code
 a = Array()
 a.reserve(size)
 @endcode

 */
FALCON_DECLARE_FUNCTION(reserve,"size:N>0");
void Function_reserve::invoke(VMContext* ctx, int32 )
{
   Item* i_size = ctx->param(0);
   int64 size;
   if( i_size == 0 || ! i_size->isOrdinal() || (size = i_size->forceInteger()) < 0 )
   {
      throw paramError(__LINE__, SRC );
   }

   ItemArray* array = static_cast<ItemArray*>(ctx->self().asInst());
   ConcurrencyGuard::Writer rg(ctx, array->guard());
   array->reserve((length_t)size);
   ctx->returnFrame(ctx->self());
}

/**
 @method clone Array
 @brief Performs a flat copy of the whole array.
 @return a new copy of the array.

 Overriding @a BOM.clone() to account for Parallel guard.

 */
FALCON_DECLARE_FUNCTION(clone,"");
void Function_clone::invoke(VMContext* ctx, int32 )
{
   Class* cls = 0;
   void* data = 0;
   ctx->self().asClassInst( cls, data );
   ItemArray* array = static_cast<ItemArray*>(data);
   ConcurrencyGuard::Writer rg(ctx, array->guard());
   void* cl = cls->clone(array);
   ctx->returnFrame( FALCON_GC_STORE(cls,cl) );
}


/*#
   @method insert Array
   @brief Inserts one or more items into this array array.
   @param pos  The position where the item should be placed.
   @param item The item to be inserted.
   @optparam ... more items to be inserted
   @return this same item.

   The item is inserted before the given position. If pos is 0, the item is
   inserted in the very first position, while if it's equal to the array length, it
   is appended at the array tail.

   If pos is negative, the position is relative to the last item, and items
   are inserted @b after the given item.

   @note When used statically, this method takes a target array as first parameter.
*/

FALCON_DECLARE_FUNCTION(insert,"array:A,pos:N,item:X,...");
void Function_insert::invoke(VMContext* ctx, int32 pCount )
{
   Item *i_array, *i_pos;
   ctx->getMethodicParams(i_array, i_pos );

   int base = ctx->isMethodic() ? 1 : 2;
   if ( i_array == 0 || ! i_array->isArray() ||
         i_pos == 0 || ! i_pos->isOrdinal() ||
         pCount <= base )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   int64 pos = i_pos->forceInteger();

   ConcurrencyGuard::Writer gw(ctx,array->guard());
   array->reserve(pCount-base + array->length() );

   if( pos < 0 )
   {
      pos = array->length() + pos + 1;
   }

   for( int i = base; i < pCount; ++i )
   {
      if ( ! array->insert( *ctx->param(i), static_cast<length_t>(pos++) ) )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC ) );
      }
   }

   ctx->returnFrame(*i_array);
}


/*#
   @method erase Array
   @brief Deletes the first element matching the given items.
   @param item The item that must be deleted.
   @optparam ... Others items to be deleted.
   @return true if at least one item has been removed, false otherwise.

   The function scans the array searching for an item that is considered
   equal to the given one(s). If such an item can be found, it is removed and
   the function returns true. If the item cannot be found, false is returned.

   This method will delete the first item matching the given one only.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(erase,"array:A,item:X,...");
void Function_erase::invoke(VMContext* ctx, int32 pCount )
{
   Item *i_array;
   ctx->getMethodicParams(i_array );

   int base = ctx->isMethodic() ? 0 : 1;
   if ( i_array == 0 || ! i_array->isArray() ||
         pCount <= base )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();

   ConcurrencyGuard::Writer gw(ctx,array->guard());

   bool bRemoved = false;
   for( int i = base; i < pCount; ++i )
   {
      int32 pos = array->find( *ctx->param(i) );
      if ( pos >= 0 )
      {
         bRemoved = true;
         array->remove(pos);
      }
   }

   ctx->returnFrame( Item().setBoolean(bRemoved) );
}


/*#
   @method eraseAll Array
   @brief Deletes the all the elements matching the given items.
   @param item The item that must be deleted.
   @optparam ... Others items to be deleted.
   @return true if at least one item has been removed, false otherwise.

   The function scans the array searching for an item that is considered
   equal to the given one(s). If such an item can be found, it is removed and
   the function returns true. If the item cannot be found, false is returned.

   This method will delete all the items matching the given ones in the array.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(eraseAll,"array:A,item:X,...");
void Function_eraseAll::invoke(VMContext* ctx, int32 pCount )
{
   Item *i_array;
   ctx->getMethodicParams(i_array );

   int base = ctx->isMethodic() ? 0 : 1;
   if ( i_array == 0 || ! i_array->isArray() ||
         pCount <= base )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();

   ConcurrencyGuard::Writer gw(ctx,array->guard());

   bool bRemoved = false;
   for( int i = base; i < pCount; ++i )
   {
      Item& current =  *ctx->param(i);
      int32 pos = array->find(current);
      while ( pos >= 0 )
      {
         bRemoved = true;
         array->remove(pos);
         pos = array->find(current);
      }
   }

   ctx->returnFrame( Item().setBoolean(bRemoved) );
}


/*#
   @method remove Array
   @brief Removes items at a given position from an array.
   @param pos Item to be deleted
   @optparam count Count of items to be delete starting at pos.

   If pos is less than zero, it will be considered relative
   to the last element in the array.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(remove,"array:A,pos:N,count:[N]");
void Function_remove::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_pos, *i_len;
   ctx->getMethodicParams(i_array, i_pos, i_len);

   if ( i_array == 0 || ! i_array->isArray()
        || i_pos == 0 || ! i_pos->isOrdinal()
        || (i_len != 0 && ! i_len->isOrdinal() )
        )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   int64 pos = i_pos->forceInteger();
   int64 len = i_len != 0 ? i_len->forceInteger() : 0;
   if( len <= 0 )
   {
      len = 1;
   }

   if( pos < 0 )
   {
      pos = array->length() + pos;
   }
   if( pos < 0 || pos >array->length() )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_arracc );
   }
   array->remove( (length_t) pos, (length_t) len);

   ctx->returnFrame();
}


/*#
   @method slice Array
   @brief Cuts the array in two parts.
   @param pos Item to be deleted
   @optparam count Number of items to be deleted
   @return A new array containing the element from pos to the end of this array.

   This method modifies the original array moving all the element from
   @b pos included to the end of the array to a new array, that is then
   returned.

   If @b pos is 0, all the elements are moved in the new array, and
   the source array is cleared.

   If @b pos is equal to the source array length, the newly returned
   array is empty and the source is unmodified.

   If @b count is given, this method will extract a part of the array,
   moving the items from @b pos to @b pos + @b count in the new array.
   If @b pos + @b count is past the end of the original array, this
   method works as if @b count was not given.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(slice,"array:A,pos:N,count:[N]");
void Function_slice::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_pos, *i_len;
   ctx->getMethodicParams(i_array, i_pos, i_len);

   if ( i_array == 0 || ! i_array->isArray()
        || i_pos == 0 || ! i_pos->isOrdinal()
        || (i_len != 0 && ! i_len->isOrdinal() )
        )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   int64 pos = i_pos->forceInteger();
   int64 len = i_len != 0 ? i_len->forceInteger() : 0;
   if( len <= 0 )
   {
      len = array->length();
   }

   if( pos < 0 )
   {
      pos = array->length() + pos;
   }

   if( pos < 0 || pos >array->length() )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_arracc );
   }

   ItemArray* retarr = new ItemArray;
   if ( pos < array->length() )
   {
      retarr->copyOnto(0, *array, pos, len );
      array->remove(pos,len);
   }

   ctx->returnFrame( FALCON_GC_HANDLE(retarr) );
}

/*#
   @method copy Array
   @brief Copy a part of the array in a new array.
   @param pos The first item to be copied.
   @optparam count Count of items to be delete starting at pos.
   @return A new array containing a flat copy of the specified
           elements.

   If @b pos is less than zero, it will be considered relative
   to the last element in the array.

   If @b count is not given, all the array from @b pos to the end
   is copied.

   @note For a complete flat copy of all elements, use @a BOM.clone

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(copy,"array:A,pos:N,count:[N]");
void Function_copy::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_pos, *i_len;
   ctx->getMethodicParams(i_array, i_pos, i_len);

   if ( i_array == 0 || ! i_array->isArray()
        || i_pos == 0 || ! i_pos->isOrdinal()
        || (i_len != 0 && ! i_len->isOrdinal() )
        )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   int64 pos = i_pos->forceInteger();
   int64 len = i_len != 0 ? i_len->forceInteger() : 0;
   if( len <= 0 )
   {
      len = 1;
   }

   if( pos < 0 )
   {
      pos = array->length() + pos;
   }

   if( pos < 0 || pos >array->length() )
   {
      throw FALCON_SIGN_ERROR(AccessError, e_arracc );
   }

   ItemArray* retarr = new ItemArray;
   retarr->copyOnto(0, *array, pos, len  );
   ctx->returnFrame( FALCON_GC_HANDLE(retarr) );
}


static void internal_push( Function* func, VMContext* ctx, int32 pCount )
{
   Item *i_array;
   ctx->getMethodicParams(i_array);

   int base = ctx->isMethodic() ? 0 : 1;
   if ( i_array == 0 || ! i_array->isArray()
        || pCount <= base )
   {
      throw func->paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   array->reserve(pCount-base + array->length() );

   for( int32 i = base; i < pCount; ++i )
   {
      array->append(*ctx->param(i));
   }

   ctx->returnFrame();
}

/*#
   @method push Array
   @brief Pushes one or more items at the end of the array.
   @param item the item to be pushed.
   @optparam ... other items to be pushed.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(push,"array:A,item:X,...");
void Function_push::invoke(VMContext* ctx, int32 pCount )
{
   internal_push( this, ctx, pCount );
}

/*#
   @method append Array
   @brief (Alias to @a Array.push) Pushes one or more items at the end of the array.
   @param item the item to be pushed.
   @optparam ... other items to be pushed.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(append,"array:A,item:X,...");
void Function_append::invoke(VMContext* ctx, int32 pCount )
{
   internal_push( this, ctx, pCount );
}

/*#
   @method unshift Array
   @brief Pushes one or more items at the beginning of the array.
   @param item the item to be pushed.
   @optparam ... other items to be pushed.

   @note If more than one item to be pushed is given,
   they are pushed in reverse order, as if calling the
   unshift() method multiple times.

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(unshift,"array:A,item:X,...");
void Function_unshift::invoke(VMContext* ctx, int32 pCount )
{
   Item *i_array;
   ctx->getMethodicParams(i_array );

   int base = ctx->isMethodic() ? 0 : 1;
   if ( i_array == 0 || ! i_array->isArray()
        || pCount <= base )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   array->reserve(pCount-base + array->length() );
   //TODO: optimize by moving all the items away in one step.
   for( int32 i = base; i < pCount; ++i )
   {
      array->prepend(*ctx->param(i));
   }

   ctx->returnFrame();
}


/*#
   @method pop Array
   @brief Remove one or more items from the end of the array (returning them)
   @optparam count Number of items to be removed.
   @return The removed item, if it's just one, or an array containing all the
         removed items.
   @raise AccessError if the array is empty

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(pop,"array:A,count:[N]");
void Function_pop::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_count;
   ctx->getMethodicParams(i_array, i_count);

   if ( i_array == 0 || ! i_array->isArray()
        || (i_count != 0 && ! i_count->isOrdinal()) )
   {
      throw paramError( __LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   int64 count = i_count != 0 ? i_count->forceInteger() : 0;
   length_t len = array->length();
   if( len == 0 )
   {
      throw FALCON_SIGN_XERROR(AccessError, e_arracc, .extra("Pop in empty array") );
   }

   if( count <= 1 ) {
      ctx->returnFrame((*array)[len-1]);
      array->remove(len-1,1);
   }
   else {
      if ( count > (int64) len )
      {
         throw FALCON_SIGN_XERROR(AccessError, e_arracc, .extra("Not enough elements for pop") );
      }

      ItemArray* toReturn = new ItemArray;
      toReturn->resize( count );
      toReturn->copyOnto(0, *array, len-count, count);
      ctx->returnFrame( FALCON_GC_HANDLE(toReturn) );

      array->remove(len-count,count);
   }
}

/*#
   @method shift Array
   @brief Remove one or more items from the beginning of the array (returning them)
   @optparam count Number of items to be removed.
   @return The removed item, if it's just one, or an array containing all the
         removed items.
   @raise AccessError if the array is empty

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(shift,"array:A,count:[N]");
void Function_shift::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_count;
   ctx->getMethodicParams(i_array, i_count);

   if ( i_array == 0 || ! i_array->isArray()
        || (i_count != 0 && ! i_count->isOrdinal()) )
   {
      throw paramError( __LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Writer gw(ctx, array->guard());

   int64 count = i_count != 0 ? i_count->forceInteger() : 0;
   length_t len = array->length();
   if( len == 0 )
   {
      throw FALCON_SIGN_XERROR(AccessError, e_arracc, .extra("Pop in empty array") );
   }

   if( count <= 1 ) {
      ctx->returnFrame((*array)[0]);
      array->remove(0,1);
   }
   else {
      if ( count > (int64) len )
      {
         throw FALCON_SIGN_XERROR(AccessError, e_arracc, .extra("Not enough elements for shift") );
      }

      ItemArray* toReturn = new ItemArray;
      toReturn->resize( count );
      toReturn->copyOnto(0, *array, 0, count);
      ctx->returnFrame( FALCON_GC_HANDLE(toReturn) );

      array->remove(0,count);
   }
}


/*#
   @method find Array
   @brief Searches for an item in the array.
   @param item The item to be searched.
   @return The position where the item is found, or -1 if not found.

   @note When used statically, this method takes a target array as first parameter.

*/
FALCON_DECLARE_FUNCTION(find,"array:A,item:X");
void Function_find::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_item;
   ctx->getMethodicParams(i_array, i_item);

   if ( i_array == 0 || ! i_array->isArray() || i_item == 0 )
   {
      throw paramError( __LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();

   int64 pos;
   {
      ConcurrencyGuard::Reader gw(ctx, array->guard());
      pos = (int64) array->find( *i_item );
   }
   ctx->returnFrame( pos );
}

/*#
   @method scan Array
   @brief Searches for an item in the array using an external method to check it.
   @param checker Code invoked to check for the item.
   @return The position at which the checker code evaluates to true.

   The @b checker parameter is evaluated and given each element as a
   paramter. When (if) it returns true, the position of the element
   that was passed to it is returned.

   @note When used statically, this method takes a target array as first parameter.

*/
FALCON_DECLARE_FUNCTION(scan,"array:A,checker:X");
void Function_scan::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_code;
   ctx->getMethodicParams(i_array, i_code);

   if ( i_array == 0 || ! i_array->isArray() || i_code == 0 )
   {
      throw paramError( __LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ConcurrencyGuard::Reader gr( ctx, array->guard());
   if( array->length() == 0 )
   {
      ctx->returnFrame(-1);
      return;
   }

   ctx->addLocals(1);
   ctx->local(0)->setInteger(0); // we'll start from 1
   ClassArray* carr = static_cast<ClassArray*>(methodOf());
   ctx->pushCode(carr->m_stepScanInvoke);
   ctx->callerLine(__LINE__+1);
   ctx->callItem(*i_code, 1, &array->at(0));
}

class PStepScanInvoke: public PStep
{
public:
   PStepScanInvoke() {apply=apply_;}
   virtual ~PStepScanInvoke() {}
   void describeTo( String& tgt ) const {tgt = "PStepScanInvoke";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      // found?
      bool isTrue = ctx->topData().isTrue();
      if( isTrue )
      {
         ctx->returnFrame(*ctx->local(0));
         return;
      }

      // no, better luck next time.
      ctx->popData();
      int64 id = ctx->local(0)->asInteger()+1;
      Item *i_array, *i_code;
      ctx->getMethodicParams(i_array, i_code);
      ItemArray *array = i_array->asArray();

      // guard from now on.
      ConcurrencyGuard::Reader gr( ctx, array->guard());
      if( id >= array->length() )
      {
         // failed.
         ctx->returnFrame(-1);
         return;
      }

      // trying with next
      ctx->local(0)->setInteger(id);
      ctx->callerLine(__LINE__+1);
      ctx->callItem( *i_code, 1, &array->at(id) );
   }
};


/*#
   @method compact Array
   @brief Reduces the memory used by an array.
   @return Itself

   Normally, array operations, as insertions, additions and so on
   cause the array to grow to accommodate items that may come in the future.
   Also, reducing the size of an array doesn't automatically dispose of the
   memory held by the array to store its elements.

   This method grants that the memory used by the array is strictly the
   memory needed to store the data that it currently contains.

   @note This will also reset the growth factor to the default minimum
   (usually 16).

   @note When used statically, this method takes a target array as first parameter.
*/
FALCON_DECLARE_FUNCTION(compact,"array:A");
void Function_compact::invoke(VMContext* ctx, int32 )
{
   Item *array_x;
   ctx->getMethodicParams(array_x);
   if ( array_x == 0 || !array_x->isArray() )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic());
   }

   {
      ConcurrencyGuard::Writer gr( ctx, array_x->asArray()->guard() );
      array_x->asArray()->compact();
      array_x->asArray()->growth(16);
   }
   ctx->returnFrame( *array_x );
}

/*#
   @method merge Array
   @brief Merges the given array into this one.
   @param source Array containing the second half of the merge, read-only
   @optparam insertPos Position of array 1 at which to place array2
   @optparam start First element of array to merge in this array
   @optparam count Count of elements of array2 to merge in this array
   @return This same array

   @note When used statically, this method takes a target array as first parameter.
*/

FALCON_DECLARE_FUNCTION(merge,"array:A,source:A,insertPos:[N],start:[N],count:[N]");
void Function_merge::invoke(VMContext* ctx, int32 )
{
   Item *first_i, *from_i, *second_i, *start_i, *end_i;
   ctx->getMethodicParams(first_i, second_i, from_i, start_i, end_i);

   if ( first_i == 0 || ! first_i->isArray()
       || second_i == 0 || ! second_i->isArray()
       || ( from_i != 0 && ! from_i->isOrdinal() )
       ||( start_i != 0 && ! start_i->isOrdinal() )
       ||( end_i != 0 && ! end_i->isOrdinal() )
       )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic());
   }

   ItemArray *first = first_i->asArray();
   ItemArray *second = second_i->asArray();
   if( first == second )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC ).extra("Same source & target arrays") );
   }

   int64 from = from_i == 0 ? first->length() : from_i->forceInteger();
   ConcurrencyGuard::Writer gw(ctx,first->guard());

   if( from < 0 )
   {
      from = first->length()+from+1;
   }

   if( from < 0 || from > first->length() )
   {
      throw paramError(__LINE__, SRC, ctx->isMethodic());
   }

   {
      ConcurrencyGuard::Reader gr(ctx,second->guard());
      int64 start = start_i == 0 ? 0 : start_i->forceInteger();
      int64 end = end_i == 0 ? second->length() : end_i->forceInteger();
      first->merge(from,*second,start,end);
   }

   ctx->returnFrame(*first_i);
}



//==========================================================================
// Sort
//

/*#
   @method sort Array
   @brief Sorts the given array in place.
   @optparam less Method to determine the ordering criteria
   @return this same array.

   If @b less is not given, the items are ordered according to the
   standard Falcon comparison operator. Strings are sorted lexicographically
   depending on the UNICODE value of their characters, items are sorted
   based on their position in memory.

   If @b less is given, it's a code that is evaluated passing two parameters,
   and it should return a boolean value indicating if the first parameter is
   "less than" the second, like the following code:

   @code
      {(a,b) a<b}
   @endcode

   @note When used statically, this method takes a target array as first parameter.
*/

static void internal_quicksort( ItemArray& array, int32 first, int32 last)
{
   int32 i = first;
   int32 j = last;
   Item pivot = array[(first + last) / 2];

   while (i <= j) {
      while ( array[i] < pivot    ) { i++; }
      while ( pivot    < array[j] ) { j--; }

      if (i <= j) {
            array[i].swap(array[j]);
            i++;
            j--;
      }
   };

   /* recursion */
   if (first < j)
      internal_quicksort(array, first, j);

   if (i < last)
      internal_quicksort(array, i, last);
}

FALCON_DECLARE_FUNCTION(sort,"array:A,less:C");
void Function_sort::invoke(VMContext* ctx, int32 )
{
   Item *array_x, *i_less;
   ctx->getMethodicParams(array_x, i_less);
   if ( array_x == 0 || !array_x->isArray() )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic());
   }

   {
      ItemArray* array = array_x->asArray();

      int32 len = static_cast<int32>(array->length());
      if( len == 0 )
      {
         ctx->returnFrame(*array_x);
         return;
      }

      ConcurrencyGuard::Writer gr( ctx, array->guard() );
      if( i_less == 0 )
      {
         internal_quicksort( *array, 0, len-1);
         ctx->returnFrame(*array_x);
         return;
      }

      ctx->pushData(array->at(len/2));
      ctx->pushData(0);  // first
      ctx->pushData( len - 1 );  // last
      // initialize the counters.
      ctx->pushData(0);  // first
      ctx->pushData( len - 1 );  // last
   }

   ClassArray* cla = static_cast<ClassArray*>(methodOf());
   ctx->pushCode( cla->m_stepQSort );
   // don't abandon the function frame.
}


class PStepQSort: public PStep
{
public:
   PStepQSort( ClassArray* ca ): m_ca(ca) {apply=apply_;}
   virtual ~PStepQSort() {}
   void describeTo( String& tgt ) const {tgt = "PStepQSort";}

   static void apply_(const PStep* ps, VMContext* ctx )
   {
      const PStepQSort* self = static_cast<const PStepQSort*>(ps);

      int32 j = (int32) ctx->opcodeParam(0).asInteger();
      int32 i = (int32) ctx->opcodeParam(1).asInteger();
      int32 last = (int32) ctx->opcodeParam(2).asInteger();
      int32 first = (int32) ctx->opcodeParam(3).asInteger();
      Item& pivot = ctx->opcodeParam(4);

      Item *array_x, *i_less;
      ctx->getMethodicParams(array_x, i_less);
      ItemArray* array = array_x->asArray();
      ConcurrencyGuard::Writer gr( ctx, array->guard() );

      if( i <= j )
      {
         // concurrency broken in the meanwhile ?
         if ( j >= (int32) array->length() )
         {
            throw FALCON_SIGN_XERROR( ConcurrencyError, e_concurrence, .extra("During Array.sort") );
         }

         ctx->pushCode(self->m_ca->m_stepQSortPartHigh);
         ctx->pushCode(self->m_ca->m_stepQSortPartLow);

         // initialize the dance
         Item params[2];
         params[0] = array->at(i);
         params[1] = pivot;
         ctx->callerLine(__LINE__+1);
         ctx->callItem(*i_less, 2, params );
         return;
      }

      ctx->popCode();
      ctx->popData(5);

      bool more = false;
      if (first < j)
      {
         ctx->pushData( array->at( (first+j)/2) );
         ctx->pushData( first );  // first
         ctx->pushData( j );  // last
         ctx->pushData( first );  // first
         ctx->pushData( j );  // last
         ctx->pushCode(self);
         more = true;
      }

      if( i < last )
      {
         ctx->pushData( array->at( (i+last)/2) );
         ctx->pushData( i );  // first
         ctx->pushData( last );  // last
         ctx->pushData( i );  // first
         ctx->pushData( last );  // last
         ctx->pushCode(self);
         more = true;
      }

      // are we done for good?
      if ( ! more ) {
         ctx->returnFrame(Item(array->handler(),array));
      }
   }

private:
   ClassArray* m_ca;
};


class PStepQSortLow: public PStep
{
public:
   PStepQSortLow() {apply=apply_;}
   virtual ~PStepQSortLow() {}
   void describeTo( String& tgt ) const {tgt = "PStepQSortLow";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      // we receive the result in top-data
      bool isLess = ctx->topData().isTrue();
      ctx->popData();

      int32 j = (int32) ctx->opcodeParam(0).asInteger();
      int32 i = (int32) ctx->opcodeParam(1).asInteger();
      Item& pivot = ctx->opcodeParam(4);

      Item *array_x, *i_less;
      ctx->getMethodicParams(array_x, i_less);
      ItemArray* array = array_x->asArray();

      ConcurrencyGuard::Writer gr( ctx, array->guard() );

      // concurrency broken in the meanwhile ?
      if ( j >= (int32) array->length() )
      {
         throw FALCON_SIGN_XERROR( ConcurrencyError, e_concurrence, .extra("During Array.sort") );
      }

      Item params[2];
      if( isLess )
      // increment the pointer and retry
      {
         i++;
         ctx->opcodeParam(1).setInteger(i);
         params[0] = array->at(i);
         params[1] = pivot;
      }
      else
      {
         // we're done
         ctx->popCode();
         // work for the High loop:
         params[0] = pivot;
         params[1] = array->at(j);
      }

      ctx->callerLine(__LINE__+1);
      ctx->callItem(*i_less, 2, params );
   }
};

class PStepQSortPartHigh: public PStep
{
public:
   PStepQSortPartHigh() {apply=apply_;}
   virtual ~PStepQSortPartHigh() {}
   void describeTo( String& tgt ) const {tgt = "PStepQSortPartHigh";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      // we receive the result in top-data
      bool isLess = ctx->topData().isTrue();
      ctx->popData();

      int32 j = (int32) ctx->opcodeParam(0).asInteger();
      int32 i = (int32) ctx->opcodeParam(1).asInteger();
      Item& pivot = ctx->opcodeParam(4);

      Item *array_x, *i_less;
      ctx->getMethodicParams(array_x, i_less);
      ItemArray* array = array_x->asArray();

      ConcurrencyGuard::Writer gr( ctx, array->guard() );
      // concurrency broken in the meanwhile ?
      if ( j >= (int32) array->length() )
      {
         throw FALCON_SIGN_XERROR( ConcurrencyError, e_concurrence, .extra("During Array.sort") );
      }

      if( isLess )
         // else, increment the pointer and retry
      {
         j--;
         ctx->opcodeParam(0).setInteger(j);
         Item params[2];
         params[0] = pivot;
         params[1] = array->at(j);

         ctx->callerLine(__LINE__+1);
         ctx->callItem(*i_less, 2, params );
      }
      else
      {
         // we're done -- let the main loop do the rest
         ctx->popCode();
         // prepare for next loop in base step
         if (i <= j) {
            array->at(i).swap(array->at(j));
            i++;
            j--;
            ctx->opcodeParam(0).setInteger(j);
            ctx->opcodeParam(1).setInteger(i);
         }
      }
   }
};


class PStepCompareNext: public PStep
{
public:
   PStepCompareNext() {apply=apply_;}
   virtual ~PStepCompareNext() {}
   void describeTo( String& tgt ) const {tgt = "PStepCompareNext";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      // op(0) top --> compare result
      // op(1) --> second operand
      // op(2) --> first operand
      int64 result = ctx->topData().asInteger();

      if( result == 0 )
      {
         // prepare the next loop
         ItemArray* arr = ctx->opcodeParam(2).asArray();
         // more result to get?
         int32 &count = ctx->currentCode().m_seqId;
         ++count;
         if( count < (int) arr->length() )
         {
            ItemArray* oarr = ctx->opcodeParam(1).asArray();
            // push (change) first operand
            ctx->topData() = arr->at(count);
            Class* cls = 0;
            void* data = 0;
            ctx->topData().forceClassInst(cls,data);
            // push second operand
            ctx->pushData( oarr->at(count) );
            // compare
            cls->op_compare(ctx,data);

            return;
         }
      }

      // remove 2 operands
      ctx->popData(2);
      ctx->topData().setInteger(result);
      ctx->popCode();
   }
};


/**
 @method fill Array
 @brief Changes all or some of the items in an array to a specified value
 @param item The item that will be replicated.
 @optparam start The first item in the array to be overwritten
 @optparam count Number of items to be overwritten

 If the @b item item is not given, all the elements in the returned array are
 set to @b nil. If it's given, each element is a clone'd entity of the given item.

@note When used statically, this method takes a target array as first parameter.
 */
FALCON_DECLARE_FUNCTION(fill,"array:A,item:X,start:[N],count:[N]");
void Function_fill::invoke(VMContext* ctx, int32 )
{
   Item *array_x, *i_item, *i_start, *i_count;
   ctx->getMethodicParams(array_x, i_item, i_start, i_count);
   if ( array_x == 0 || !array_x->isArray()
      || i_item == 0
      || (i_start != 0 && ! i_start->isOrdinal() )
      || (i_count != 0 && ! i_count->isOrdinal() )
   )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic());
   }

   ItemArray* array = array_x->asArray();
   ConcurrencyGuard::Writer gr( ctx, array->guard() );

   int32 len = static_cast<int32>(array->length());
   if( len == 0 )
   {
      ctx->returnFrame(*array_x);
      return;
   }

   int64 start = i_start == 0 ? 0 : i_start->forceInteger();
   int64 count = i_count == 0 ? len : i_count->forceInteger() ;
   if( start < 0 )
   {
      start = len - start;
   }

   if( start < 0 || start >= len )
   {
      throw FALCON_SIGN_XERROR( AccessError, e_arracc, .extra("start out of range") );
   }

   if( count < 0 || (start + count > len) )
   {
      count = len - start;
   }

   internal_fill( array, i_item, start, count );

   ctx->returnFrame(*array_x);
}

/**
 @method resize Array
 @brief Resize the array to the given size.
 @param size The item that will be replicated.
 @optparam dflt Default value to be used to fill new items in the array.
 @return This same array.

 This method enlarges, eventually reallocating, or shrink an existing array.

 If @b size is zero, then all the items are removed from the array.

 If @b size is less than zero, then the size of the array is reduced of
 the absolute value of that size. For instance, if it's -3, then the
 size of the array is reduced by 3.

 If @b size is greater than the current size, the array is enlarged. If the
 @b item is not given, all the new elements in array are
 set to @b nil. If it's given, each new element is a clone'd
 entity of the given item.

 @note When used statically, this method takes a target array as first parameter.
 */
FALCON_DECLARE_FUNCTION(resize,"array:A,size:[N],dflt:[X]");
void Function_resize::invoke(VMContext* ctx, int32 )
{
   Item *array_x, *i_size, *i_dflt;
   ctx->getMethodicParams(array_x, i_size, i_dflt);
   if ( array_x == 0 || !array_x->isArray()
      || i_size == 0 || ! i_size->isOrdinal()
   )
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic());
   }

   ItemArray* array = array_x->asArray();
   ConcurrencyGuard::Writer gr( ctx, array->guard() );

   int32 len = static_cast<int32>(array->length());
   int64 size = i_size->asInteger();
   if( size != len )
   {
      if( size < 0 )
      {
         size = len + size;
      }
      if( size < 0 ) size = 0;

      array->resize(size);

      if( size > len && i_dflt != 0 )
      {
         internal_fill(array, i_dflt, len, size-len);
      }
   }

   ctx->returnFrame(*array_x);
}


/**
 @method clear Array
 @brief Removes all the items from this array
 @return This same array.
 @note When used statically, this method takes a target array as first parameter.
 */
FALCON_DECLARE_FUNCTION(clear,"array:A");
void Function_clear::invoke(VMContext* ctx, int32 )
{
   Item *array_x;
   ctx->getMethodicParams(array_x);
   if ( array_x == 0 || !array_x->isArray())
   {
      throw paramError(__LINE__,SRC, ctx->isMethodic());
   }

   ItemArray* array = array_x->asArray();
   ConcurrencyGuard::Writer gr( ctx, array->guard() );
   array->resize(0);
   ctx->returnFrame(*array_x);
}

/*#
   @method sieve Array
   @brief Removes elements not matching a criterion in place.
   @param checker Code invoked to check for items to be filtered.
   @return this same array.

   The @b checker parameter is evaluated and given each element as a
   paramter. If it returns true, the item is kept, otherwise it's removed
   from the array, and the array is resized accordingly.

   This uses the same source array, and if the original array is not needed,
   it results in being more efficient than the @a filter() generic function,
   which creates a new array copying there the items for which the check
   is true.
*/
FALCON_DECLARE_FUNCTION(sieve,"checker:C");
void Function_sieve::invoke(VMContext* ctx, int32 )
{
   Item *i_code = ctx->param(0);

   if ( i_code == 0 )
   {
      throw paramError( __LINE__, SRC );
   }

   ItemArray *array = ctx->tself<ItemArray>();
   ConcurrencyGuard::Reader gr( ctx, array->guard());
   if( array->length() == 0 )
   {
      ctx->returnFrame(ctx->self());
      return;
   }

   ctx->addLocals(1);
   ctx->local(0)->setInteger(0); // we'll start from 1
   ClassArray* carr = static_cast<ClassArray*>(methodOf());
   ctx->pushCode(carr->m_stepSieveNext);
   ctx->callerLine(__LINE__+1);
   ctx->callItem(*i_code, 1, &array->at(0));
}

class PStepSieveNext: public PStep
{
public:
   PStepSieveNext() { apply = apply_; }
   virtual ~PStepSieveNext() {}
   virtual void describeTo(String& tgt) const {
      tgt = "Array::PStepSieveNext";
   }

   static void apply_(const PStep*, VMContext* ctx )
   {
      ItemArray *array = ctx->tself<ItemArray>();
      int64 id = ctx->local(0)->asInteger();
      bool opRes = ctx->topData().isTrue();

      // what's the result of the sieve?
      ConcurrencyGuard::Writer gr( ctx, array->guard());
      if ( ! opRes )
      {
         array->remove(id);
      }
      else {
         ctx->local(0)->setInteger(++id);
      }

      if( static_cast<length_t>(id) >= array->length() )
      {
         ctx->returnFrame(ctx->self());
      }
      else
      {
         Item *i_code = ctx->param(0);
         ctx->callItem(*i_code, 1, &array->at(id));
      }
   }
};

}

//==========================================================================================
// Main class
//
ClassArray::ClassArray():
   Class("Array", FLC_CLASS_ID_ARRAY )
{
   setConstuctor( new _classArray::Function_init);
   addProperty( "len", &get_len );
   addProperty( "empty", &get_empty );
   addProperty( "allocated", &get_allocated );
   addProperty( "growth", &get_growth, &set_growth );

   // pure static functions
   addMethod( new _classArray::Function_alloc, true );
   addMethod( new _classArray::Function_buffer, true );

   // Methodic functions
   addMethod( new _classArray::Function_compact, true );

   addMethod( new _classArray::Function_insert, true );
   addMethod( new _classArray::Function_remove, true );
   addMethod( new _classArray::Function_erase, true );
   addMethod( new _classArray::Function_eraseAll, true );

   addMethod( new _classArray::Function_push, true );
   addMethod( new _classArray::Function_append, true );
   addMethod( new _classArray::Function_pop, true );
   addMethod( new _classArray::Function_shift, true );
   addMethod( new _classArray::Function_unshift, true );

   addMethod( new _classArray::Function_find, true );
   addMethod( new _classArray::Function_scan, true );

   addMethod( new _classArray::Function_copy, true );
   addMethod( new _classArray::Function_slice, true );
   addMethod( new _classArray::Function_merge, true );
   addMethod( new _classArray::Function_sort, true );

   addMethod( new _classArray::Function_fill, true );
   addMethod( new _classArray::Function_resize, true );
   addMethod( new _classArray::Function_clear, true );

   // Non-methodic functions
   addMethod( new _classArray::Function_reserve );
   addMethod( new _classArray::Function_clone );
   addMethod( new _classArray::Function_sieve );

   m_stepScanInvoke = new _classArray::PStepScanInvoke;

   m_stepQSort = new _classArray::PStepQSort(this);
   m_stepQSortPartLow = new _classArray::PStepQSortLow;
   m_stepQSortPartHigh = new _classArray::PStepQSortPartHigh;
   m_stepCompareNext = new _classArray::PStepCompareNext;

   m_stepSieveNext = new _classArray::PStepSieveNext;
}


ClassArray::~ClassArray()
{
   delete m_stepScanInvoke;

   delete m_stepQSort;
   delete m_stepQSortPartLow;
   delete m_stepQSortPartHigh;
   delete m_stepCompareNext;

   delete m_stepSieveNext;
}

int64 ClassArray::occupiedMemory( void* instance ) const
{
   ItemArray* f = static_cast<ItemArray*>( instance );
   return sizeof(ItemArray) + (f->allocated() * sizeof(Item)) + 16 + (f->allocated()?16:0);
}


void ClassArray::dispose( void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   delete array;
}


void* ClassArray::clone( void* source ) const
{
   ItemArray* array = static_cast<ItemArray*>( source );

   return new ItemArray( *array );
}


void ClassArray::store( VMContext*, DataWriter* stream, void* instance ) const
{
   // ATM we have the growth parameter to save.
   ItemArray* arr = static_cast<ItemArray*>( instance );
   stream->write( arr->m_growth );
}


void ClassArray::restore( VMContext* ctx, DataReader* stream ) const
{
   uint32 growth;

   stream->read( growth );

   ItemArray* iarr = new ItemArray;

   iarr->m_growth = growth;
   ctx->pushData( Item(this, iarr) );
}


void ClassArray::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   // ItemArray* self = static_cast<ItemArray*>( instance );

   subItems.replicate( *( static_cast<ItemArray*>( instance ) ) );
}


void ClassArray::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ItemArray* self = static_cast<ItemArray*>( instance );

   self->replicate( subItems );
}


void ClassArray::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   ItemArray* arr = static_cast<ItemArray*>( instance );

   if(maxDepth == 0)
   {
      target.append("[...]");
      return;
   }

   target.append( "[" );

   String temp;

   for ( length_t pos = 0; pos < arr->length(); ++ pos )
   {
      if ( pos > 0 )
      {
         target.append( ", " );
      }

      Class* cls;
      void* inst;

      arr->at( pos ).forceClassInst( cls, inst );

      temp.size( 0 );

      cls->describe( inst, temp, maxDepth - 1, maxLen );

      target.append( temp );
   }

   target.append( "]" );
}

void* ClassArray::createInstance() const
{
   return new ItemArray;
}


bool ClassArray::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   ItemArray* array = static_cast<ItemArray*>(instance);
   if ( pcount > 0 )
   {
      array->resize( pcount );
      Item* first = ctx->opcodeParams( pcount );
      for( int pos = 0; pos < pcount; ++ pos )
      {
         (*array)[pos] = first[pos];
      }
   }

   return false;
}



void ClassArray::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *arritem;

   ctx->operands( arritem, index );

   ItemArray& array = *static_cast<ItemArray*>( self );

   if ( index->isOrdinal())
   {
      int64 v = index->forceInteger();

      if( v < 0 ) v = array.length() + v;

      if( v < 0 || v >= array.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "out of range" ) );
      }

      ctx->stackResult( 2, array[(length_t)v] );
   }
   else if ( index->isUser() ) // index is a range
   {
      // if range is moving from a smaller number to larger (start left move right in the array)
      //      give values in same order as they appear in the array
      // if range is moving from a larger number to smaller (start right move left in the array)
      //      give values in reverse order as they appear in the array

      Class *cls;
      void *udata;

      if ( ! index->asClassInst( cls, udata ) )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "type error" ) );
      }

      if ( cls->typeID() == FLC_CLASS_ID_RANGE )
      {
         Range& rng = *static_cast<Range*>( udata );

         int64 step = ( rng.step() == 0 ) ? 1 : rng.step(); // No step given assume 1
         int64 start = rng.start();
         int64 end = rng.end();
         int64 arrayLen = array.length();

         // do some validation checks before proceeding
         if ( start >= arrayLen || start < ( arrayLen * -1 )  || end > arrayLen || end < ( arrayLen * -1 ) )
         {
            throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
         }

         bool reverse = false;

         if ( rng.isOpen() )
         {
            // If negative number start from the end of the array
            if ( start < 0 ) start = arrayLen + start;

            end = arrayLen;
         }
         else
         {
            if ( start < 0 ) start = arrayLen + start;

            if ( end < 0 ) end = arrayLen + end;

            if ( start > end )
            {
               reverse = true;
               if ( rng.step() == 0 ) step = -1;
            }
         }

         ItemArray *returnArray = new ItemArray( abs( (int)(( start - end ) + 1) ) );

         if ( reverse )
         {
            while ( start >= end )
            {
               returnArray->append( array[(length_t)start] );
               start += step;
            }
         }
         else
         {
            while ( start < end )
            {
               returnArray->append( array[(length_t)start] );
               start += step;
            }
         }

         ctx->stackResult( 2, FALCON_GC_STORE( this, returnArray ) );
      }
      else
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "unknown index" ) );
      }
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "invalid index" ) );
   }
}


void ClassArray::op_setIndex( VMContext* ctx, void* self ) const
{
   // arritem[index] = value
   // arritem also = self
   Item* value, *arritem, *index;
   ctx->operands( value, arritem, index );

   ItemArray& array = *static_cast<ItemArray*>( self );

   if ( index->isOrdinal() )
   {
      int64 v = index->forceInteger();

      if( v < 0 ) v = array.length() + v;

      if( v < 0 || v >= array.length() )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
      }

      array[(length_t)v].copyFromLocal(*value);
      ctx->popData(2); // our value is already the thirdmost element in the stack.
      return;
   }
   else if ( ! index->isUser() ) // should get a user type
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ) );
   }

   Class *rangeClass, *rhs;
   void *udataRangeInst, *udataRhs;

   if ( ! index->asClassInst( rangeClass, udataRangeInst ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "range type error" ) );
   }

   if ( rangeClass->typeID() != FLC_CLASS_ID_RANGE )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "unknown index" ) );
   }

   Range& rng = *static_cast<Range*>( udataRangeInst );

   int64 arrayLen = array.length();
   int64 start = rng.start();
   int64 end = ( rng.isOpen() ) ? arrayLen : rng.end();


   // handle negative number indexs
   if ( start < 0 ) start = arrayLen + start;
   if ( end < 0 ) end = arrayLen + end;

   int64 rangeLen = ( rng.isOpen() ) ? arrayLen - start : end - start;

   // do some validation checks before proceeding
   if ( start >= arrayLen || start < ( arrayLen * -1 )  || end > arrayLen || end < ( arrayLen * -1 ) )
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__ ).extra( "index out of range" ) );
   }

   if ( value->asClassInst( rhs, udataRhs ) ) // is the rhs a deep class
   {
      if ( rhs->typeID() == FLC_CLASS_ID_ARRAY )
      {
         ItemArray& rhsArray = *static_cast<ItemArray*>( udataRhs );

         int64 rhsLen = rhsArray.length();

         if ( rangeLen < rhsLen )
         {
            array.change( rhsArray, (length_t)start, (length_t)rangeLen );
         }
         else if ( rangeLen > rhsLen )
         {
            array.change( rhsArray, (length_t)start, (length_t)rhsLen );

            array.remove( (length_t)( start + rhsLen ), (length_t)( rangeLen - rhsLen ) );
         }
         else // rangeLen == rhsLen
         {
            array.copyOnto( (length_t)start, rhsArray, 0, (length_t)rhsLen );
         }
      }
      else
      {
         array[ (length_t)(( start == end ) ? start + 1 : start) ] = *value;

         if ( rangeLen > 1 )
         {
            array.remove( (length_t)( start + 1 ), (length_t)( rangeLen - 1 ) );
         }
      }
   }
   else  // Not a deep class
   {
      if ( rangeLen > 1 )
      {
         array[ (length_t)(( start == end ) ? start + 1 : start) ] = *value;

         array.remove( (length_t)( start + 1 ), (length_t)( rangeLen - 1 ) );
      }
      else  // arritem[1:1]
      {
         array.insert( *value, (Falcon::length_t) start );
      }
   }

   // our value is already the third element in the stack.
   ctx->popData( 2 );
}


void ClassArray::gcMarkInstance( void* self, uint32 mark ) const
{
   ItemArray& array = *static_cast<ItemArray*>( self );
   array.gcMark( mark );
}



//=======================================================================
//

void ClassArray::op_add( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item *op2 = &ctx->topData();

   ItemArray *result = new ItemArray;
   // we might throw
   FALCON_GC_HANDLE( result );
   result->reserve( array->length() + 1 );
   {
      ConcurrencyGuard::Reader gr(ctx,array->guard());
      result->merge( *array );
   }

   result->append( *op2 );
   ctx->stackResult( 2, Item( result->handler(), result ) );
}


void ClassArray::op_aadd( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );

   {
      ConcurrencyGuard::Writer gr(ctx,array->guard());
      array->append( ctx->topData() );
   }

   // just remove the topmost item,
   ctx->popData();
}


void ClassArray::op_sub( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item *op2 = &ctx->topData();

   ItemArray *result = new ItemArray;
   // we might throw
   FALCON_GC_HANDLE( result );
   result->reserve( array->length() + 1 );
   {
      ConcurrencyGuard::Reader gr(ctx,array->guard());
      result->merge( *array );
   }

   int32 pos = result->find(*op2);
   if( pos >= 0 )
   {
      result->remove(pos);
   }

   ctx->stackResult( 2, Item( result->handler(), result ) );
}


void ClassArray::op_asub( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );

   {
      ConcurrencyGuard::Writer gr(ctx,array->guard());
      int32 pos = array->find(ctx->topData());
      if( pos >= 0 )
      {
         array->remove(pos);
      }
   }

   // just remove the topmost item,
   ctx->popData();
}

void ClassArray::op_shl( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item& top = ctx->topData();

   // a basic type?
   Class* cls;
   void* inst;
   if( ! top.asClassInst( cls, inst ) || ! cls->isDerivedFrom(this) )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("A") );
   }

   // it's an array!
   ItemArray *result = new ItemArray;
   // we might throw...
   FALCON_GC_HANDLE(result);
   ItemArray* other = static_cast<ItemArray*>( cls->getParentData(this,inst) );

   {
      ConcurrencyGuard::Reader gr1(ctx,array->guard());
      ConcurrencyGuard::Reader gr2(ctx,other->guard());
      result->reserve( array->length() + other->length() );
      result->merge( *array );
      result->merge( *other );
   }

   ctx->popData();
   ctx->topData().setUser(result->handler(), result);

}

void ClassArray::op_ashl( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item& top = ctx->topData();

   // a basic type?
   Class* cls;
   void* inst;
   if( ! top.asClassInst( cls, inst ) || ! cls->isDerivedFrom(this) )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("A") );
   }

   // it's an array!
   ItemArray* other = static_cast<ItemArray*>( cls->getParentData(this,inst) );
   if( other == array )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("Auto-op on itself") );
   }

   {
      ConcurrencyGuard::Writer gr1(ctx,array->guard());
      ConcurrencyGuard::Reader gr2(ctx,other->guard());
      array->merge( *other );
   }

   ctx->popData();
}


void ClassArray::op_shr( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item& top = ctx->topData();

   // a basic type?
   Class* cls;
   void* inst;
   if( ! top.asClassInst( cls, inst ) || ! cls->isDerivedFrom(this) )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("A") );
   }

   // it's an array!
   ItemArray *result = new ItemArray;
   // we might throw below
   FALCON_GC_HANDLE( result );
   ItemArray* other = static_cast<ItemArray*>( cls->getParentData(this,inst) );

   {
      ConcurrencyGuard::Reader gr1(ctx,array->guard());
      result->merge( *array );
   }

   {
      ConcurrencyGuard::Reader gr2(ctx,other->guard());
      for( length_t pos = 0; pos < other->length(); ++pos )
      {
         Item& elem = (*other)[pos];
         int32 posl = result->find(elem);
         if( posl >= 0 )
         {
            result->remove(posl);
         }
      }
   }

   ctx->popData();
   ctx->topData().setUser(result->handler(), result);
}

void ClassArray::op_ashr( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   Item& top = ctx->topData();

   // a basic type?
   Class* cls;
   void* inst;
   if( ! top.asClassInst( cls, inst ) || ! cls->isDerivedFrom(this) )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("A") );
   }

   // it's an array!
   ItemArray* other = static_cast<ItemArray*>( cls->getParentData(this,inst) );
   if( other == array )
   {
      throw new OperandError( ErrorParam(e_op_params, __LINE__, SRC ).extra("Auto-op on itself") );
   }

   {
      ConcurrencyGuard::Writer gr1(ctx,array->guard());
      ConcurrencyGuard::Reader gr2(ctx,other->guard());
      for( length_t pos = 0; pos < other->length(); ++pos )
      {
         Item& elem = (*other)[pos];
         int32 posl = array->find(elem);
         if( posl >= 0 )
         {
            array->remove(posl);
         }
      }
   }

   ctx->popData();
}


void ClassArray::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, Item().setBoolean(static_cast<ItemArray*>( self )->length() != 0) );
}


void ClassArray::op_in( VMContext* ctx, void* instance ) const
{
   Item *item, *index;
   ctx->operands( item, index );

   bool res;
   ItemArray& array = *static_cast<ItemArray*>(instance);
   {
      ConcurrencyGuard::Reader rg( ctx, array.guard());
      res = array.find( *index ) >= 0;
   }

   ctx->popData();
   ctx->topData().setBoolean(res);
}


void ClassArray::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Array of ").N( static_cast<ItemArray*>( self )->length() ).A( " elements]" );
   ctx->stackResult( 1, s );
}


void ClassArray::op_call( VMContext* ctx, int32 paramCount, void* self ) const
{
   ItemArray* data = static_cast<ItemArray*>( self );
   length_t len = data->length();

   if( len == 0 )
   {
      Class::op_call( ctx, paramCount, self );
   }
   else
   {
      Item* elems = data->elements();

      Class *cls;
      void* udata;
      elems->forceClassInst( cls, udata );
      if( udata == data )
      {
         // infinite expand loop.
         throw new CodeError( ErrorParam( e_call_loop, __LINE__, SRC )
            .origin( ErrorParam::e_orig_vm )
            );
      }

      // expand the array as parameters.
      ctx->insertData( paramCount + 1, data->elements(), (int32) len, 1 );

      // invoke the call.
      cls->op_call( ctx, len + paramCount-1, udata );
   }
}


void ClassArray::op_iter( VMContext* ctx, void* ) const
{
   // (seq)
   ctx->pushData( (int64) 0 );
}


void ClassArray::op_next( VMContext* ctx, void* instance ) const
{
   // (seq)(iter)
   Item& iter = ctx->opcodeParam( 0 );
   length_t pos = 0;

   ItemArray* arr = static_cast<ItemArray*>( instance );

   fassert( iter.isInteger() );
   pos = (length_t) iter.asInteger();

   if( pos >= arr->length() )
   {
      ctx->addDataSlot().setBreak();
   }
   else
   {
      Item& value = arr->at( pos++ );
      iter.setInteger( pos );
      ctx->pushData( value );

      if( pos != arr->length() )
      {
         ctx->topData().setDoubt();
      }
   }
}


void ClassArray::op_compare( VMContext* ctx, void* instance ) const
{
   ItemArray* arr = static_cast<ItemArray*>( instance );
   Item& other = ctx->topData();
   int64 cfr = typeID() - other.type();
   if( cfr != 0 )
   {
      ctx->popData();
      ctx->topData().setInteger(cfr);
      return;
   }

   ItemArray* oarr = other.asArray();
   cfr = ((int64)arr->length()) -((int64) oarr->length());
   if( cfr != 0 )
   {
      ctx->popData();
      ctx->topData().setInteger(cfr);
      return;
   }


   // the two arrays MIGHT be equal.
   if( arr->length() == 0 )
   {
      ctx->popData();
      // indeed they are.
      ctx->topData().setInteger(0);
   }
   else {
      // ready to check next compare.
      ctx->pushCode( m_stepCompareNext );

      ctx->pushData(arr->at(0));
      ctx->pushData(oarr->at(0));
      Class* cls = 0;
      void* data;
      arr->at(0).forceClassInst(cls,data);
      cls->op_compare(ctx,data);
   }
}


}

/* end of classarray.cpp */

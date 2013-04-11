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

#include <falcon/errors/accesserror.h>
#include <falcon/errors/codeerror.h>
#include <falcon/errors/paramerror.h>

namespace Falcon {

static void get_len( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<ItemArray*>( instance )->length();
}

static void get_empty( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean(static_cast<ItemArray*>( instance )->length() == 0);
}

namespace _classArray {
/**
 @class Array
 @brief Class of Falcon language array []
 @param size Initial size of the array.
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
      array->resize(len);

      if( i_dflt != 0 )
      {
         Class* cls = 0;
         void* data = 0;
         if( i_dflt->asClassInst(cls, data) )
         {
            for( length_t pos = 0; pos < (length_t) len; ++pos )
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

   ctx->returnFrame();
}

/**
 @method reserve Array
 @brief Allocates more memory to store new data without resizing the array.
 @param size The size that must be preallocated.

 The memory occupied by this array is enlarged to the required size
 so that the array that is guaranteed not to be reallocated up to at least
 @b size items are added to it.

 If the size is smaller than the current array size, nothing is done.

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
   ctx->returnFrame();
}


/*#
   @method insert Array
   @brief Inserts one or more items into this array array.
   @param pos  The position where the item should be placed.
   @param item The item to be inserted.
   @optparam ... more items to be inserted

   The item is inserted before the given position. If pos is 0, the item is
   inserted in the very first position, while if it's equal to the array length, it
   is appended at the array tail.

   If pos is negative, the position is relative to the last item, and items
   are inserted @b after the given item.
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
      if ( ! array->insert( *ctx->param(i), pos++ ) )
      {
         throw new AccessError( ErrorParam( e_arracc, __LINE__, SRC ) );
      }
   }

   ctx->returnFrame();
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
*/
FALCON_DECLARE_FUNCTION(remove,"array:A,pos:N,count:[N]");
void Function_remove::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_pos, *i_len;
   ctx->getMethodicParams(i_array, i_pos, i_len);

   if ( i_array == 0 || ! i_array->isArray()
        || i_pos == 0 || ! i_pos->isOrdinal()
        || (i_len == 0 && ! i_len->isOrdinal() )
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
   @method push Array
   @brief Pushes one or more items at the end of the array.
   @param item the item to be pushed.
   @optparam ... other items to be pushed.
*/
FALCON_DECLARE_FUNCTION(push,"array:A,item:X,...");
void Function_push::invoke(VMContext* ctx, int32 pCount )
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

   for( int32 i = base; i < pCount; ++i )
   {
      array->append(*ctx->param(i));
   }

   ctx->returnFrame();
}

/*#
   @method unshift Array
   @brief Pushes one or more items at the beginning of the array.
   @param item the item to be pushed.
   @optparam ... other items to be pushed.

   @note If more than one item to be pushed is given,
   they are pushed in reverse order, as if calling the
   unshift() method multiple times.
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

   int64 count = i_count->forceInteger();
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
         count = len;
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

   int64 count = i_count->forceInteger();
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
         count = len;
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

*/
FALCON_DECLARE_FUNCTION(find,"array:A,item:[N]");
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

*/
FALCON_DECLARE_FUNCTION(scan,"array:A,item:[N]");
void Function_scan::invoke(VMContext* ctx, int32 )
{
   Item *i_array, *i_code;
   ctx->getMethodicParams(i_array, i_code);

   if ( i_array == 0 || ! i_array->isArray() || i_code == 0 )
   {
      throw paramError( __LINE__, SRC, ctx->isMethodic() );
   }

   ItemArray *array = i_array->asArray();
   ctx->addLocals(2);
   ctx->local(0)->setInteger(0);
   ctx->local(1)->setInteger(array->length());
   ClassArray* carr = static_cast<ClassArray*>(array->handler());
   ctx->stepIn(carr->m_stepScanInvoke);
}

class PStepScanInvoke: public PStep
{
public:
   PStepScanInvoke() {apply=apply_;}
   virtual ~PStepScanInvoke() {}
   void describeTo(String& tgt) {tgt = "PStepScanInvoke";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      Item *i_array, *i_code;
      ctx->getMethodicParams(i_array, i_code);
      ItemArray *array = i_array->asArray();

      Item item;
      {
         ConcurrencyGuard::Reader gr( ctx, array->guard());
         item = (*array)[ctx->local(0)->asInteger()];
      }

      ClassArray* carr = static_cast<ClassArray*>(array->handler());
      ctx->pushCode(carr->m_stepScanCheck);

      ctx->callItem( *i_code, 1, &item );
   }
};

class PStepScanCheck: public PStep
{
public:
   PStepScanCheck() {apply=apply_;}
   virtual ~PStepScanCheck() {}
   void describeTo(String& tgt) {tgt = "PStepScanCheck";}

   static void apply_(const PStep*, VMContext* ctx )
   {
      int64 pos = ctx->local(0)->asInteger();

      if( ctx->topData().isBoolean() && ctx->topData().asBoolean() )
      {
         ctx->returnFrame(pos);
      }
      else {
         if( ++pos == ctx->local(1)->asInteger() )
         {
            // failed...
            ctx->returnFrame((int64)-1);
            return;
         }
         // we can try again
         ctx->local(0)->setInteger(pos);
         ctx->popCode();
      }
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
   memory needed to store its data.
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
   }
   ctx->returnFrame( *array_x );
}

/*#
   @method merge Array
   @brief Merges the given array into this one.
   @param insertPos Position of array 1 at which to place array2
   @param source Array containing the second half of the merge, read-only
   @optparam start First element of array to merge in this array
   @optparam end Last element of array2 to merge in this array

   @see arrayMerge
*/

FALCON_DECLARE_FUNCTION(merge,"array:A,insertPos:N,source:A,start:[N],end:[N]");
void Function_merge::invoke(VMContext* ctx, int32 )
{
   Item *first_i, *from_i, *second_i, *start_i, *end_i;
   ctx->getMethodicParams(first_i, from_i, second_i, start_i, end_i);

   if ( first_i == 0 || ! first_i->isArray()
       || from_i == 0 || ! from_i->isOrdinal()
       || second_i == 0 || ! second_i->isArray()
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

   int64 from = from_i->forceInteger();
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
      first->copyOnto(from,*second,start,end);
   }

   ctx->returnFrame();
}


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

   // pure static functions
   addMethod( new _classArray::Function_alloc, true );
   addMethod( new _classArray::Function_buffer, true );

   // Methodic functions
   addMethod( new _classArray::Function_insert, true );
   addMethod( new _classArray::Function_remove, true );
   addMethod( new _classArray::Function_erase, true );
   addMethod( new _classArray::Function_eraseAll, true );

   addMethod( new _classArray::Function_push, true );
   addMethod( new _classArray::Function_pop, true );
   addMethod( new _classArray::Function_shift, true );
   addMethod( new _classArray::Function_unshift, true );

   addMethod( new _classArray::Function_find, true );
   addMethod( new _classArray::Function_scan, true );

   addMethod( new _classArray::Function_compact, true );
   addMethod( new _classArray::Function_merge, true );

   // Non-methodic functions
   addMethod( new _classArray::Function_reserve );

   m_stepScanInvoke = new _classArray::PStepScanInvoke;
   m_stepScanCheck = new _classArray::PStepScanCheck;
}


ClassArray::~ClassArray()
{
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

      if( v >= array.length() )
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

      if( v >= array.length() )
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


void ClassArray::enumerateProperties( void*, Class::PropertyEnumerator& ) const
{
   // TODO array bindings?
}

void ClassArray::enumeratePV( void*, Class::PVEnumerator& ) const
{
   // TODO array bindings?
}

//=======================================================================
//

void ClassArray::op_add( VMContext* ctx, void* self ) const
{
   static Class* arrayClass = Engine::handlers()->arrayClass();

   ItemArray* array = static_cast<ItemArray*>( self );
   Item* op1, *op2;
   ctx->operands( op1, op2 );

   Class* cls;
   void* inst;

   ItemArray *result = new ItemArray;

   // a basic type?
   if( ! op2->asClassInst( cls, inst ) || cls->typeID() != typeID() )
   {
      result->reserve( array->length() + 1 );
      result->merge( *array );
      result->append( *op2 );
   }
   else {
      // it's an array!
      ItemArray* other = static_cast<ItemArray*>( inst );

      result->reserve( array->length() + other->length() );
      result->merge( *array );
      result->merge( *other );
   }

   ctx->stackResult( 2, Item( FALCON_GC_STORE( arrayClass, result ) ) );
}


void ClassArray::op_aadd( VMContext* ctx, void* self ) const
{
   ItemArray* array = static_cast<ItemArray*>( self );
   array->append( ctx->topData() );

   // just remove the topmost item,
   ctx->popData();
}


void ClassArray::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, static_cast<ItemArray*>( self )->length() != 0 );
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


}

/* end of classarray.cpp */

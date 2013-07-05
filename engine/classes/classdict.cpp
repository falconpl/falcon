/*
   FALCON - The Falcon Programming Language.
   FILE: classdict.cpp

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 15:33:32 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classdict.cpp"


#include <falcon/classes/classdict.h>
#include <falcon/itemid.h>
#include <falcon/vmcontext.h>
#include <falcon/itemdict.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/stdhandlers.h>

#include <falcon/stderrors.h>

namespace Falcon {

static void get_len( const Class*, const String&, void* instance, Item& value )
{
   value = (int64) static_cast<ItemDict*>( instance )->size();
}

static void get_empty( const Class*, const String&, void* instance, Item& value )
{
   value.setBoolean( static_cast<ItemDict*>( instance )->size() == 0 );
}

static void get_pseudo( const Class*, const String&, void* instance, Item& value )
{
   static Class* cls = Engine::instance()->stdHandlers()->pseudoDictClass();
   ItemDict* dict = static_cast<ItemDict*>( instance );

   value.setUser(cls,dict);
}



namespace _classDictionary {

/**
 @class Dictionary
 @brief Class of Falcon language dictionary [ => ]
 @param ... Items to be initially stored in the dictionary

 The items passed to the constructor must be in pair number; odd elements
 are the key for the subsequent even element (the first if the key to the
 second, the third to the fourth and so on).

 @note At the moment, lexicographic ordering is granted only for numeric, range
 and string values, other than the @b nil, @b true and @b false predefined values. 
 Other objects are not inserted as keys in the dictionary checking them against
 their compare() method, but thrugh the exactly equal operator ===. This might
 change in a future release.

 @section dict_ops Overloaded operators.

 Other than the obvious subscript access operators ([] and []=), the following
 operators are managed by this class:

 - add(+): clones this dictionary and adds an array of paired elements
          in an array or dictionary to it.
 - aadd(+=): inserts a set of paired elements in an array or in a dictionary.
 - sub(-): clones this dictionary and removes exactly one key from that, if found.
 - asub(-): removes exactly one key from this dictionary, if found.
 - shl(<<): alias for add(+)
 - ashl(<<=): alias for aadd(+=)
 - shr(>>): Clones this dictionary and then removes all the keys found in the array on the right
           of the operators.
 - ashr(>>=): Removes all the keys found in the array on the right of the operator from this array.
 - in: Checks if a key is present in this dictionary.

 @prop len Count of key/value pairs in the dictionary
 @prop empty true if there isn't any element in the dictionary
*/

/**
 @method keys Dictionary
 @brief Returns an array containing all the keys in the dictionary.
 @return An array containing all the keys in the dictionary.

 If the dictionary is empty, this method returns an empty array.
 */
FALCON_DECLARE_FUNCTION(keys,"");
void Function_keys::invoke(VMContext* ctx, int32)
{
   // we need to enumerate all the keys/values in the array ...
   class FlatterEnum: public ItemDict::Enumerator {
   public:
      FlatterEnum( ItemArray& tgt ):
         m_tgt( tgt )
      {}
      virtual ~FlatterEnum(){}

      virtual void operator()( const Item& key, Item& )
      {
         m_tgt.append( key );
      }

   private:
      ItemArray& m_tgt;
   };

   ItemArray* array = new ItemArray;

   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Reader rg(ctx, dict->guard() );
   array->reserve( dict->size() );

   FlatterEnum rator( *array );
   dict->enumerate(rator);

   ctx->returnFrame( FALCON_GC_HANDLE(array) );
}


/**
 @method values Dictionary
 @brief Returns an array containing all the values in the dictionary.
 @return An array containing all the values in the dictionary.

 If the dictionary is empty, this method returns an empty array.
 */
FALCON_DECLARE_FUNCTION(values,"");
void Function_values::invoke(VMContext* ctx, int32)
{
   // we need to enumerate all the keys/values in the array ...
   class FlatterEnum: public ItemDict::Enumerator {
   public:
      FlatterEnum( ItemArray& tgt ):
         m_tgt( tgt )
      {}
      virtual ~FlatterEnum(){}

      virtual void operator()( const Item&, Item& value )
      {
         m_tgt.append( value );
      }

   private:
      ItemArray& m_tgt;
   };

   ItemArray* array = new ItemArray;

   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Reader rg(ctx, dict->guard() );
   array->reserve( dict->size() );

   FlatterEnum rator( *array );
   dict->enumerate(rator);

   ctx->returnFrame( FALCON_GC_HANDLE(array) );
}


/**
 @method clone Dictionary
 @brief Returns a flat copy of this dictionary.
 @return A flat copy of this dictionary.

 */
FALCON_DECLARE_FUNCTION(clone,"");
void Function_clone::invoke(VMContext* ctx, int32)
{
   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Reader rg(ctx, dict->guard() );
   ctx->returnFrame( FALCON_GC_HANDLE(dict->clone()) );
}

/**
 @method clear Dictionary
 @brief Removes all the elements from this dictionary.
 @return This same dictionary.

 */
FALCON_DECLARE_FUNCTION(clear,"");
void Function_clear::invoke(VMContext* ctx, int32)
{
   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Writer wr(ctx, dict->guard() );
   dict->clear();
   ctx->returnFrame( Item( dict->handler(), dict ) );
}

/**
 @method find Dictionary
 @brief Searches for an entity in the dictionary.
 @param key The key item to be found.
 @optparam dflt The value returned if the item is not found.
 @return The value associated with the found key, or @b dflt if not found.
 @raise AccessError if the key is not found and dflt is not given.
 
*/
FALCON_DECLARE_FUNCTION(find,"key:X,dflt:[X]");
void Function_find::invoke(VMContext* ctx, int32)
{
   Item* i_key = ctx->param(0);
   if( i_key == 0 )
   {
      throw paramError();
   }
   Item* i_dflt = ctx->param(1);

   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Reader rg( ctx, dict->guard() );
   
   Item* found = dict->find(*i_key);
   if( found == 0 )
   {
      if( i_dflt == 0 )
      {
         throw FALCON_SIGN_ERROR(AccessError, e_arracc ); 
      }
      ctx->returnFrame( *i_dflt );
   }
   else {
      ctx->returnFrame( *found );
   }
}


/**
 @method remove Dictionary
 @brief Removes an entity from the dictionary.
 @param key The key item to be removed. 
 @return true if the item was removed, false otherwise.

 If the key is not found, this method does nothing.
*/
FALCON_DECLARE_FUNCTION(remove,"key:X");
void Function_remove::invoke(VMContext* ctx, int32)
{
   Item* i_key = ctx->param(0);
   if( i_key == 0 )
   {
      throw paramError();
   }  

   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Writer wg( ctx, dict->guard() );
   bool b = dict->remove(*i_key);
   ctx->returnFrame(Item().setBoolean(b));
}

/**
 @method insert Dictionary
 @brief Inserts an entity from the dictionary
 @param key The key item to be removed. 
 @param value The key item to be removed. 

 Adds a new key-value pair to the dictionary, or overwrites
 it if the pair was already existing.
*/
FALCON_DECLARE_FUNCTION(insert,"key:X,value:X");
void Function_insert::invoke(VMContext* ctx, int32)
{
   Item* i_key = ctx->param(0);
   Item* i_value = ctx->param(1);
   if( i_key == 0 || i_value == 0 )
   {
      throw paramError();
   }  

   ItemDict* dict = static_cast<ItemDict*>(ctx->self().asInst());
   ConcurrencyGuard::Writer wg( ctx, dict->guard() );
   dict->insert(*i_key, *i_value);
   ctx->returnFrame();
}

}


ClassDict::ClassDict():
   Class("Dictionary", FLC_CLASS_ID_DICT )
{
   init();
}


ClassDict::ClassDict( const String& subclsName ):
   Class( subclsName, FLC_CLASS_ID_DICT )
{
   init();
}


void ClassDict::init()
{
   addProperty( "len", &get_len );
   addProperty( "empty", &get_empty );
   addProperty( "pseudo", &get_pseudo );

   // pure static functions
   addMethod( new _classDictionary::Function_keys );
   addMethod( new _classDictionary::Function_values );
   addMethod( new _classDictionary::Function_clone );
   addMethod( new _classDictionary::Function_clear );

   addMethod( new _classDictionary::Function_find );
   addMethod( new _classDictionary::Function_remove );
   addMethod( new _classDictionary::Function_insert );
}


ClassDict::~ClassDict()
{
}


void ClassDict::dispose( void* self ) const
{
   ItemDict* f = static_cast<ItemDict*>(self);
   delete f;
}


void* ClassDict::clone( void* source ) const
{
   return static_cast<ItemDict*>(source)->clone();
}

void* ClassDict::createInstance() const
{
   return new ItemDict;
}

void ClassDict::store( VMContext*, DataWriter* stream, void* instance ) const
{
   ItemDict* dict = static_cast<ItemDict*>(instance);
   stream->write(dict->m_flags);
   stream->write(dict->m_version);
   
}


void ClassDict::restore( VMContext* ctx, DataReader* stream ) const
{
   // first read the data (which may throw).
   uint32 flags, version;
   stream->read(flags);
   stream->read(version);
   
   // when we're done, create the entity.
   ItemDict* dict = new ItemDict;
   dict->m_flags = flags;
   dict->m_version = flags;
   ctx->pushData( Item( this, dict) );
}


void ClassDict::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   // we need to enumerate all the keys/values in the array ...
   class FlatterEnum: public ItemDict::Enumerator {
   public:
      FlatterEnum( ItemArray& tgt ):
         m_tgt( tgt )
      {}
      virtual ~FlatterEnum(){}
      
      virtual void operator()( const Item& key, Item& value )
      {
         m_tgt.append( key );
         m_tgt.append( value );
      }
      
   private:
      ItemArray& m_tgt;
   };
   
   FlatterEnum rator( subItems );
   
   // However, we have at least an hint about the enumeration size.
   ItemDict* dict = static_cast<ItemDict*>(instance);
   subItems.reserve( dict->size() * 2 );
   dict->enumerate(rator);
   
}


void ClassDict::unflatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   ItemDict* dict = static_cast<ItemDict*>(instance);
   uint32 size = subItems.length();
   if( size %2  != 0 )
   {
      // woops something wrong.
      throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
         .origin( ErrorParam::e_orig_runtime )
         .extra( "Unmatching keys/values"));
   }
   
   for(uint32 i = 0; i < size; i += 2 )
   {
      dict->insert( subItems[i], subItems[i+1] );
   }
}


void ClassDict::describe( void* instance, String& target, int maxDepth, int maxLen ) const
{
   if( maxDepth == 0 )
   {
      target = "...";
      return;
   }

   ItemDict* dict = static_cast<ItemDict*>(instance);
   dict->describe(target, maxDepth, maxLen);
}



void ClassDict::gcMarkInstance( void* self, uint32 mark ) const
{
   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.gcMark( mark );
}


//=======================================================================
//
bool ClassDict::op_init( VMContext* ctx, void* inst, int pCount ) const
{
   // TODO: create the dictionary
   if( pCount % 2 != 0 )
   {
      throw FALCON_SIGN_XERROR(ParamError, e_param_arity, .extra("Must be multiple of 2"));
   }

   ItemDict* dict = static_cast<ItemDict*>(inst);
   Item* items = ctx->opcodeParams(pCount);
   for(int32 i = 0; i < pCount; i+=2)
   {
      dict->insert(items[i], items[i+1]);
   }
   return false;
}


void ClassDict::op_add( VMContext* ctx, void* inst ) const
{
   ItemDict* self_dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   ItemDict* dict = 0;
   if( top.isArray() )
   {
      ItemArray& arr = *ctx->topData().asArray();
      ConcurrencyGuard::Reader rd( ctx, arr.guard() );
      length_t len = arr.length();
      if( len % 2 != 0 )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Array length must be multiple of 2"));
      }

      ConcurrencyGuard::Reader rg1( ctx, self_dict->guard() );
      dict = self_dict->clone();
      for( length_t i = 0; i < len; i+=2 )
      {
         dict->insert(arr[i], arr[i+1]);
      }

   }
   else if( top.isDict() )
   {
      ItemDict* second = ctx->topData().asDict();
      // merge with myself does nothing.
      if( second == dict )
      {
         ctx->popData();
         return;
      }

      ConcurrencyGuard::Reader rg1( ctx, self_dict->guard() );
      ConcurrencyGuard::Reader rd( ctx, second->guard() );
      dict = self_dict->clone();
      dict->merge( *second );
   }
   else {
      throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Need to be an Array of Dictionary"));
   }

   ctx->popData();
   ctx->topData() = FALCON_GC_HANDLE(dict);
}


void ClassDict::op_aadd( VMContext* ctx, void* inst ) const
{
   ItemDict* dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   if( top.isArray() )
   {
      ItemArray& arr = *ctx->topData().asArray();
      ConcurrencyGuard::Reader rd( ctx, arr.guard() );
      length_t len = arr.length();
      if( len % 2 != 0 )
      {
         throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Array length must be multiple of 2"));
      }

      ConcurrencyGuard::Writer wr( ctx, dict->guard() );
      for( length_t i = 0; i < len; i+=2)
      {
         dict->insert(arr[i], arr[i+1]);
      }
   }
   else if( top.isDict() )
   {
      ItemDict* second = ctx->topData().asDict();
      // merge with myself does nothing.
      if( second == dict )
      {
         ctx->popData();
         return;
      }

      ConcurrencyGuard::Writer wr( ctx, dict->guard() );
      ConcurrencyGuard::Reader rd( ctx, second->guard() );
      dict->merge( *second );
   }
   else {
      throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Need to be an Array of Dictionary"));
   }

   ctx->popData();
}


void ClassDict::op_sub( VMContext* ctx, void* inst ) const
{
   ItemDict* self_dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   ConcurrencyGuard::Reader rg( ctx, self_dict->guard() );
   ItemDict* dict = self_dict->clone();
   dict->remove(top);
   ctx->popData();
   ctx->topData() = FALCON_GC_HANDLE(dict);
}


void ClassDict::op_asub( VMContext* ctx, void* inst ) const
{
   ItemDict* self_dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   ConcurrencyGuard::Writer rg( ctx, self_dict->guard() );
   self_dict->remove(top);
   ctx->popData();
}


void ClassDict::op_shl( VMContext* ctx, void* inst ) const
{
   op_add( ctx, inst );
}

void ClassDict::op_ashl( VMContext* ctx, void* inst ) const
{
   op_add( ctx, inst );
}


void ClassDict::op_shr( VMContext* ctx, void* inst ) const
{
   ItemDict* self_dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   ItemDict* dict = 0;
   if( top.isArray() )
   {
      ItemArray& arr = *ctx->topData().asArray();
      ConcurrencyGuard::Writer wr( ctx, self_dict->guard() );
      dict = self_dict->clone();

      ConcurrencyGuard::Reader rd( ctx, arr.guard() );
      for( length_t i = 0; i < arr.length(); ++i )
      {
         dict->remove(arr[i]);
      }
   }
   else {
      throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Need to be an Array"));
   }

   ctx->popData();
   ctx->topData() = FALCON_GC_HANDLE(dict);
}

void ClassDict::op_ashr( VMContext* ctx, void* inst ) const
{
   ItemDict* dict = static_cast<ItemDict*>(inst);
   Item& top = ctx->topData();

   if( top.isArray() )
   {
      ItemArray& arr = *ctx->topData().asArray();
      ConcurrencyGuard::Writer wr( ctx, dict->guard());
      ConcurrencyGuard::Reader rd( ctx, arr.guard() );
      for( length_t i = 0; i < arr.length(); ++i )
      {
         dict->remove(arr[i]);
      }
   }
   else {
      throw FALCON_SIGN_XERROR(ParamError, e_op_params, .extra("Need to be an Array of Dictionary"));
   }

   ctx->popData();
}


void ClassDict::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, static_cast<ItemDict*>(self)->size() != 0 );
}


void ClassDict::op_in( VMContext* ctx, void* instance ) const
{
   Item *item, *index;
   ctx->operands( item, index );

   bool res;
   ItemDict& dict = *static_cast<ItemDict*>(instance);
   {
      ConcurrencyGuard::Reader rg( ctx, dict.guard());
      res = dict.find( *index ) != 0;
   }

   ctx->popData();
   ctx->topData().setBoolean(res);
}


void ClassDict::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Dictionary of ").N((int64)static_cast<ItemDict*>(self)->size()).A(" elements]");
   ctx->stackResult( 1, s );
}


void ClassDict::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *item, *index;
   ctx->operands( item, index );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   ConcurrencyGuard::Reader wr( ctx, dict.guard());
   Item* result = dict.find( *index );

   if( result != 0 )
   {
      ctx->stackResult( 2, *result );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__, __FILE__ ) );
   }
}

void ClassDict::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *item, *index;
   ctx->operands( value, item, index );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   ConcurrencyGuard::Writer wr( ctx, dict.guard());
   dict.insert( *index, *value );
   ctx->stackResult(3, *value);
}

void ClassDict::op_iter( VMContext* ctx, void* instance ) const
{
   static Class* genc = Engine::handlers()->genericClass();
   
   ItemDict* dict = static_cast<ItemDict*>(instance);
   ItemDict::Iterator* iter = new ItemDict::Iterator( dict );
   ctx->pushData( FALCON_GC_STORE( genc, iter ) );
}


void ClassDict::op_next( VMContext* ctx, void*  ) const
{
   Item& user = ctx->opcodeParam( 0 );
   fassert( user.isUser() );
   fassert( user.asClass() == Engine::handlers()->genericClass() );
   
   ItemDict::Iterator* iter = static_cast<ItemDict::Iterator*>(user.asInst());
   ctx->addSpace(1);
   ConcurrencyGuard::Reader rd( ctx, iter->dict()->guard() );
   if( ! iter->next( ctx->topData() ) )
   {
      throw new AccessError(  ErrorParam(e_async_seq_modify, __LINE__, SRC) );
   }
}
}

/* end of classdict.cpp */

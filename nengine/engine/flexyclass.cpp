/*
   FALCON - The Falcon Programming Language.
   FILE: flexyclass.cpp

   Class handling flexible objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 18 Jul 2011 02:22:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/flexyclass.h>
#include <falcon/itemdict.h>
#include <falcon/item.h>

#include <falcon/vmcontext.h>
#include <falcon/accesserror.h>
#include <falcon/engine.h>

namespace Falcon
{


FlexyClass::FlexyClass():
   OverridableClass( "Flexy" )
{
}


FlexyClass::~FlexyClass()
{
}


void FlexyClass::dispose( void* self ) const
{
   delete static_cast<ItemDict*>(self);
}


void* FlexyClass::clone( void* self ) const
{
   return static_cast<ItemDict*>(self)->clone();
}


void FlexyClass::serialize( DataWriter*, void*  ) const
{
   // TODO
}


void* FlexyClass::deserialize( DataReader* ) const
{
   // TODO
   return 0;
}


void FlexyClass::gcMark( void* self, uint32 mark ) const
{
   static_cast<ItemDict*>(self)->gcMark(mark);
}


void FlexyClass::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   class Enum: public ItemDict::Enumerator
   {
   public:
      Enum( Class::PropertyEnumerator& cb ):
         m_cb(cb)
      {}

      virtual void operator()( const Item& key, Item& )
      {
         if( key.isString() )
         {
            const String* prop = const_cast<Item*>(&key)->asString();
            if( prop->find(" ") == String::npos )
            {
               m_cb( *prop, false );
            }
         }

      }
      
   private:
      Class::PropertyEnumerator& m_cb;
   };

   Enum rator(cb);
   static_cast<ItemDict*>(self)->enumerate( rator );
}


bool FlexyClass::hasProperty( void* self, const String& prop ) const
{
   static Class* scls = Engine::instance()->stringClass();
   
   Item key( scls, const_cast<String*>(&prop) );  // create a static string.
   return static_cast<ItemDict*>(self)->find( key ) != 0;
}


void FlexyClass::describe( void* self, String& target, int depth, int maxlen ) const
{
   class Enum: public ItemDict::Enumerator
   {
   public:
      Enum( String& target, int depth, int maxlen ):
         m_depth(depth),
         m_maxlen( maxlen ),
         m_target( target )
      {}

      virtual void operator()( const Item& key, Item& value )
      {
         if( key.isString() )
         {
            const String* prop = const_cast<Item*>(&key)->asString();
            if( prop->find(" ") == String::npos )
            {
               if( m_target.size() > 1 )
               {
                  m_target += ", ";
               }

               m_value.size(0);
               value.describe( m_value, m_depth-1, m_maxlen );
               m_target += *prop + " => " + m_value;
            }
         }

      }

   private:
      int m_depth;
      int m_maxlen;
      String& m_target;
      String m_value;
   };

   String tgt;
   Enum rator( tgt, depth, maxlen );
   static_cast<ItemDict*>(self)->enumerate( rator );
   target.size(0);
   target += "Flexy{" + tgt + "}";
}


void FlexyClass::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();
   
   ctx->stackResult( pcount, FALCON_GC_STORE(coll, this, new ItemDict) );
}


void FlexyClass::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *index, *dict_item;
   ctx->operands( index, dict_item );

   ItemDict& dict = *static_cast<ItemDict*>(self);
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


void FlexyClass::op_setIndex( VMContext* ctx, void* self ) const
{
   Item *value, *index, *dict_item;
   ctx->operands( value, index, dict_item );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.insert( *index, *value );
   ctx->stackResult(3, *value);
}


void FlexyClass::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   static Class* scls = Engine::instance()->stringClass();

   ItemDict& dict = *static_cast<ItemDict*>(self);
   Item key( scls, const_cast<String*>(&prop) );  // create a static string.
   Item* result = dict.find( key );

   if( result != 0 )
   {
      ctx->stackResult( 2, *result );
   }
   else
   {
      throw new AccessError( ErrorParam( e_arracc, __LINE__, __FILE__ ) );
   }
}


void FlexyClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   ItemDict& dict = *static_cast<ItemDict*>(self);
   Item key = new String(prop);  // create a static string.
   dict.insert( key, ctx->opcodeParam(1) );
   ctx->stackResult(2, ctx->opcodeParam(1) );
}

}

/* end of flexyclass.cpp */

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

#define SRC "engine/flexyclass.cpp"
#include <falcon/flexydict.h>

#include <falcon/flexyclass.h>
#include <falcon/item.h>
#include <falcon/itemdict.h>

#include <falcon/vmcontext.h>
#include <falcon/accesserror.h>
#include <falcon/engine.h>

#include <map>


namespace Falcon
{

FlexyClass::FlexyClass():
   OverridableClass( "Flexy" )
{
}

FlexyClass::FlexyClass( const String& name ):
   OverridableClass( name )
{
}



FlexyClass::~FlexyClass()
{
}


void FlexyClass::dispose( void* self ) const
{
   delete static_cast<FlexyDict*>(self);
}


void* FlexyClass::clone( void* self ) const
{
   return new FlexyDict( *static_cast<FlexyDict*>(self));
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
   static_cast<FlexyDict*>(self)->gcMark(mark);
}


void FlexyClass::enumerateProperties( void* self, PropertyEnumerator& cb ) const
{
   static_cast<FlexyDict*>(self)->enumerateProps( cb );
}

void FlexyClass::enumeratePV( void* self, PVEnumerator& cb ) const
{
   static_cast<FlexyDict*>(self)->enumeratePV( cb );
}

bool FlexyClass::hasProperty( void* self, const String& prop ) const
{
   return static_cast<FlexyDict*>(self)->hasProperty( prop );
}


void FlexyClass::describe( void* self, String& target, int depth, int maxlen ) const
{
   String tgt;
   if( depth != 0 )
   {
      static_cast<FlexyDict*>(self)->describe( tgt, depth, maxlen );
      target.size(0);
      target += name() + "{" + tgt + "}";
   }
   else
   {
      tgt = name() + "{...}";
   }
}


void FlexyClass::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();

   FlexyDict* self = new FlexyDict;
   // In case of a single parameter...
   if( pcount >= 1 )
   {
      //... it can be a dictionary or a generic class.
      Item& param = ctx->opcodeParam(0);
      Class* cls;
      void* data;
      param.forceClassInst( cls, data );

      if( cls->typeID() == FLC_CLASS_ID_DICT )
      {
         class Enum: public ItemDict::Enumerator
         {
         public:
            Enum( const FlexyClass* owner, FlexyDict* self):
               m_owner(owner),
               m_self(self),
               m_fself( owner, self)
            {
               m_fself.garbage();
            }

            virtual void operator()( const Item& key, Item& value )
            {
               if( key.isString() )
               {
                  if( key.asString()->find( " " ) == String::npos )
                  {
                     if( value.isFunction() )
                     {
                        Function* func = value.asFunction();
                        Item temp = m_fself;
                        temp.methodize( func );
                        m_self->insert( *key.asString(), temp );
                     }
                     else
                     {
                        m_self->insert( *key.asString(), value );
                     }
                  }
               }
            }

         private:
            const FlexyClass* m_owner;
            FlexyDict* m_self;
            Item m_fself;
         };

         Enum rator( this, self );
         ItemDict& id = *static_cast<ItemDict*>( data );
         id.enumerate( rator );
      }
      else
      {
         class Enum: public Class::PVEnumerator
         {
         public:
            Enum( const FlexyClass* owner, FlexyDict* self):
               m_owner(owner),
               m_self(self),
               m_fself( owner, self)
            {
               m_fself.garbage();
            }

            virtual void operator()( const String& data, Item& value )
            {
               if( value.isFunction() )
               {
                  Function* func = value.asFunction();
                  Item temp = m_fself;
                  temp.methodize( func );
                  m_self->insert( data, temp );
               }
               else
               {
                  m_self->insert( data, value );
               }
            }
            
         private:
            const FlexyClass* m_owner;
            FlexyDict* m_self;
            Item m_fself;
         };

         Enum rator( this, self );
         cls->enumeratePV( data, rator );
      }

   }

   ctx->stackResult( pcount+1, FALCON_GC_STORE(coll, this, self ) );
}


void FlexyClass::op_getProperty( VMContext* ctx, void* self, const String& prop) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);
   Item* result = dict.find( prop );

   if( result != 0 )
   {
      ctx->topData() = *result; // should be already copied by insert
   }
   else
   {
      // fall back to the starndard system
      Class::op_getProperty( ctx, self, prop );
   }
}


void FlexyClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);

   ctx->popData();
   Item& value = ctx->topData();
   if ( value.isFunction() )
   {
      Function* func = value.asFunction();
      value.setUser( this, &dict, true );
      value.methodize( func );
   }
   
   dict.insert( prop, value );
}

}

/* end of flexyclass.cpp */

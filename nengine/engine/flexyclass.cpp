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
#include <falcon/item.h>
#include <falcon/itemdict.h>

#include <falcon/vmcontext.h>
#include <falcon/accesserror.h>
#include <falcon/engine.h>

#include <map>


namespace Falcon
{

class FlexyDict
{
public:
   typedef std::map<String, Item> ItemMap;
   ItemMap m_im;

   uint32 m_currentMark;
   uint32 m_flags;

   inline FlexyDict():
      m_currentMark(0),
      m_flags(0)
   {}

   inline FlexyDict( const FlexyDict& other ):
      m_currentMark(other.m_currentMark),
      m_flags(other.m_flags)
   {
      m_im = other.m_im;
   }

   inline ~FlexyDict()
   {}

   inline void gcMark( uint32 mark )
   {
      if( m_currentMark == mark )
      {
         return;
      }

      ItemMap::iterator pos = m_im.begin();
      while( pos != m_im.end() )
      {
         const Item& value = pos->second;

         if( value.isUser() && value.isGarbaged() )
         {
            value.asClass()->gcMark(value.asInst(), mark);
         }

         ++pos;
      }
   }

   inline void enumerateProps( Class::PropertyEnumerator& e ) const
   {
      ItemMap::const_iterator pos = m_im.begin();
      while( pos != m_im.end() )
      {
         const String& key = pos->first;
         e( key, ++pos == m_im.end() );
      }
   }

   inline void enumeratePV( Class::PVEnumerator& e )
   {
      ItemMap::iterator pos = m_im.begin();
      while( pos != m_im.end() )
      {
         const String& key = pos->first;
         e( key, pos->second );
         ++pos;
      }
   }

   inline bool hasProperty( const String& p ) const
   {
      return m_im.find(p) != m_im.end();
   }

   inline void describe( String& target, int depth, int maxlen ) const
   {
      String value;
      ItemMap::const_iterator pos = m_im.begin();
      while( pos != m_im.end() )
      {
         const String& key = pos->first;

         if( target.size() > 1 )
         {
            target += ", ";
         }

         value.size(0);
         pos->second.describe( value, depth-1, maxlen );
         target += key + "=" + value;

         ++pos;
      }
   }

   inline Item* find( const String& value )
   {
      ItemMap::iterator pos = m_im.find( value );
      if( pos != m_im.end() )
      {
         return &pos->second;
      }
      return 0;
   }

   inline void insert( const String& key, Item& value )
   {
      value.copied( true );
      m_im[key] = value;
   }
};

FlexyClass::FlexyClass():
   OverridableClass( "Flexy" )
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
   static_cast<FlexyDict*>(self)->describe( tgt, depth, maxlen );
   target.size(0);
   target += "Flexy{" + tgt + "}";
}


void FlexyClass::op_create( VMContext* ctx, int32 pcount ) const
{
   static Collector* coll = Engine::instance()->collector();

   FlexyDict* self = new FlexyDict;
   // In case of a single parameter...
   if( pcount == 1 )
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
            Enum(FlexyDict* self):
               m_self(self)
            {}

            virtual void operator()( const Item& key, Item& value )
            {
               if( key.isString() )
               {
                  if( key.asString()->find( " " ) == String::npos )
                  {
                     m_self->insert( *key.asString(), value );
                  }
               }
            }

         private:
            FlexyDict* m_self;
         };

         Enum rator( self );
         ItemDict& id = *static_cast<ItemDict*>( data );
         id.enumerate( rator );
      }
      else
      {
         class Enum: public Class::PVEnumerator
         {
         public:
            Enum(FlexyDict* self):
               m_self(self)
            {}

            virtual void operator()( const String& data, Item& value )
            {
               m_self->insert( data, value );
            }
            
         private:
            FlexyDict* m_self;
         };

         Enum rator( self );
         cls->enumeratePV( data, rator );
      }

   }

   ctx->stackResult( pcount, FALCON_GC_STORE(coll, this, self ) );
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
      throw new AccessError( ErrorParam( e_arracc, __LINE__, __FILE__ ) );
   }
}


void FlexyClass::op_setProperty( VMContext* ctx, void* self, const String& prop ) const
{
   FlexyDict& dict = *static_cast<FlexyDict*>(self);

   ctx->popData();
   Item& value = ctx->topData();
   dict.insert( prop, value );
}

}

/* end of flexyclass.cpp */

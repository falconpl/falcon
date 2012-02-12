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

#include <falcon/errors/accesserror.h>
#include <falcon/errors/unserializableerror.h>

namespace Falcon {


namespace DictProperties {

//
// Class properties used for enumeration
//
const int NUM_OF_PROPERTIES   = 20;

char properties[ ][ 12 ] = {
   "back",  // 1
   "best",
   "clear",
   "comp",
   "do",
   "dop",
   "fill",
   "find",
   "first",
   "front", // 10
   "get",
   "keys",
   "last",
   "mcomp",
   "merge",
   "mfcomp",
   "properties",
   "remove",
   "setProperty",
   "values" // 20
};

}


ClassDict::ClassDict():
   Class("Dictionary", FLC_CLASS_ID_DICT )
{
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


void ClassDict::restore( VMContext*, DataReader* stream, void*& empty ) const
{
   // first read the data (which may throw).
   uint32 flags, version;
   stream->read(flags);
   stream->read(version);
   
   // when we're done, create the entity.
   ItemDict* dict = new ItemDict;
   dict->m_flags = flags;
   dict->m_version = flags;
   empty = dict;
}


void ClassDict::flatten( VMContext*, ItemArray& subItems, void* instance ) const
{
   // we need to enumerate all the keys/values in the array ...
   class FlatterEnum: public ItemDict::Enumerator {
   public:
      FlatterEnum( ItemArray& tgt ):
         m_tgt( tgt )
      {}
      
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



void ClassDict::gcMark( void* self, uint32 mark ) const
{
   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.gcMark( mark );
}

void ClassDict::enumerateProperties( void*, PropertyEnumerator& cb ) const
{
   for ( int cnt = 0; cnt < ( DictProperties::NUM_OF_PROPERTIES - 1 ); cnt++ )
   {
      cb( DictProperties::properties[ cnt ], false );
   }

   cb( DictProperties::properties[ DictProperties::NUM_OF_PROPERTIES - 1 ], true );
}

void ClassDict::enumeratePV( void*, Class::PVEnumerator& ) const
{
   // EnumeratePV doesn't normally return static methods.
}

//=======================================================================
//
bool ClassDict::op_init( VMContext* ctx, void* instance, int pcount ) const
{
   // TODO: create the dictionary
   return false;
}

void ClassDict::op_add( VMContext*, void* ) const
{
   //TODO
}

void ClassDict::op_isTrue( VMContext* ctx, void* self ) const
{
   ctx->stackResult( 1, static_cast<ItemDict*>(self)->size() != 0 );
}

void ClassDict::op_toString( VMContext* ctx, void* self ) const
{
   String s;
   s.A("[Dictionary of ").N((int64)static_cast<ItemDict*>(self)->size()).A(" elements]");
   ctx->stackResult( 1, s );
}


void ClassDict::op_getProperty( VMContext* ctx, void* self, const String& property ) const
{
   Class::op_getProperty( ctx, self, property );
}

void ClassDict::op_getIndex( VMContext* ctx, void* self ) const
{
   Item *item, *index;
   ctx->operands( item, index );

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

void ClassDict::op_setIndex( VMContext* ctx, void* self ) const
{
   Item* value, *item, *index;
   ctx->operands( value, item, index );

   ItemDict& dict = *static_cast<ItemDict*>(self);
   dict.insert( *index, *value );
   ctx->stackResult(3, *value);
}

void ClassDict::op_iter( VMContext* ctx, void* instance ) const
{
   static Collector* coll = Engine::instance()->collector();
   static Class* genc = Engine::instance()->genericClass();
   
   ItemDict* dict = static_cast<ItemDict*>(instance);
   ItemDict::Iterator* iter = new ItemDict::Iterator( dict );
   ctx->pushData( FALCON_GC_STORE( coll, genc, iter ) );
}


void ClassDict::op_next( VMContext* ctx, void*  ) const
{
   Item& user = ctx->opcodeParam( 0 );
   fassert( user.isUser() );
   fassert( user.asClass() == Engine::instance()->genericClass() );
   
   ItemDict::Iterator* iter = static_cast<ItemDict::Iterator*>(user.asInst());
   ctx->addSpace(1);
   iter->next( ctx->topData() );
}

}

/* end of classdict.cpp */

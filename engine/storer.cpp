/*
   FALCON - The Falcon Programming Language.
   FILE: storer.cpp

   Helper for cyclic joint structure serialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Oct 2011 16:06:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/storer.h>
#include <falcon/class.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/gclock.h>
#include <falcon/vmcontext.h>

#include <map>
#include <vector>

namespace Falcon {

class Storer::Private
{
public:
   ItemArray* m_theArray;
   GCLock* m_lock;
   
   /** Map connecting each class with its own serialization ID. */
   typedef std::map<Class*, size_t> ClassMap;
   
   /** Map connecting each each object with its own serialization ID. */
   typedef std::map<void*, size_t> ObjectMap;   
   
   /** Vector of numeric IDs on which an object depends. */
   typedef std::vector<size_t> IDVector;
   
   /** Data relative to a signle serialized item. */
   class ObjectData {
   public:
      void* m_data;
      size_t m_id;
      size_t m_clsId;
      
      /** Items to be given back while deserializing. */
      IDVector m_deps;
      
      ObjectData():
         m_data(0),
         m_id(0),
         m_clsId(0)
      {}
      
      ObjectData( void* data, size_t id,  size_t cls ):
         m_data(data),
         m_id(id),
         m_clsId(cls)
      {}
      
      ObjectData( const ObjectData& other ):
         m_data( other.m_data ),
         m_id( other.m_id ),
         m_clsId( other.m_clsId )
      // ignore m_deps
      {}
   };
   
   typedef std::vector<ObjectData> ObjectDataVector;
   typedef std::vector<ObjectData*> ObjectDataPtrVector;
   typedef std::vector<Class*> ClassVector;
   
   ClassMap m_classes;
   ClassVector m_clsVector;
   
   ObjectMap m_objects;
   ObjectDataVector m_objVector;
   IDVector m_objBoundaries;
   ObjectDataPtrVector m_objTraversing;
   
   // used during serialization.
   uint32 m_partStart;
   uint32 m_partEnd;
   
   Private() {
      static Collector* coll = Engine::instance()->collector();
      
      m_theArray = new ItemArray;
      Item itx;
      itx.setArray( m_theArray, true );
      m_lock = coll->lock( itx );
   }
   
   ~Private() {
      m_lock->dispose();
   }
   
   void setNextObject()
   {
      m_objBoundaries.push_back( m_objVector.size() );
   }
   
   
   ObjectData* addObject( Class* cls, void* obj, bool& bIsNew )
   {
      // if the object is alredy there...
      ObjectMap::const_iterator objIter = m_objects.find(obj);
      if( objIter != m_objects.end() )
      {
         //... we have already it all
         bIsNew = false;
         return &m_objVector[objIter->second];
      }
      
      bIsNew = true;
      size_t objCount = m_objects.size()+1;
      m_objects[obj] = objCount;
      
      // ... then, see if we need to save the class as well..
      size_t clsId;
      ClassMap::const_iterator clsIter = m_classes.find(cls);
      if( clsIter != m_classes.end() )
      {
         // class was already there, get the ID
         clsId = clsIter->second;
      }
      else
      {
         // New class in town...
         clsId = m_classes.size()+1;
         m_classes[cls] = clsId;
         m_clsVector.push_back( cls );
      }
      
      m_objVector.push_back( ObjectData( obj, objCount, clsId ) );
      return &m_objVector.back();
   }
   
   inline void addTraversing( ObjectData* obj )
   {
      m_objTraversing.push_back( obj );
   }
};

//===========================================================
//
//

Storer::Storer( VMContext* ctx ):
   _p(0),
   m_ctx(ctx),
   m_traverseNext( this ),
   m_writeNext(this),
   m_writeNextPart( this )
{  
}


Storer::~Storer()
{
   delete _p;
}

   
bool Storer::store( Class* handler, void* data )
{
   if( _p == 0 )
   {
      _p = new Private;
   }
   
   // First, create the Private that we use to unroll cycles.
   return traverse( handler, data );
}


bool Storer::commit( DataWriter* wr )
{
   try
   {
      // Then store them.
      writeClassDict( wr );

      // Serialize each item.
      return doSerialize( wr );
   }
   catch( ... )
   {
      throw;
   }
}


bool Storer::traverse( Class* handler, void* data )
{
   // first, save the item.
   bool bIsNew;
   Storer::Private::ObjectData* objd = _p->addObject( handler, data, bIsNew );
   _p->addTraversing( objd );
   
   if( ! bIsNew )
   {
      //... was already there -- then we have nothing to do.
      return true;
   }
   
   
   // if the item is new, traverse its dependencies.
   m_ctx->pushCode( &m_traverseNext );
   handler->flatten( m_ctx, *_p->m_theArray, data );
   if ( m_ctx->currentCode().m_step != &m_traverseNext )
   {
      // going deep? -- suspend processing
      return false;
   }
   
   // continue processing.
   m_traverseNext.apply_( &m_traverseNext, m_ctx ); 
   return _p->m_objTraversing.empty();
}


void Storer::TraverseNext::apply_( const PStep* ps, VMContext* ctx )
{
   const TraverseNext* self = static_cast<const TraverseNext*>( ps );
   // get the object we're working on.
   Private::ObjectDataPtrVector& traversing = self->m_owner->_p->m_objTraversing;
   fassert( ! traversing.empty() );
   Private::ObjectData* objd = traversing.back();
   ItemArray& items = *self->m_owner->_p->m_theArray;
   
   int &i = ctx->currentCode().m_seqId;
   // The dependencies are now stored in items array.
   while( i < (int) items.length() ) 
   {
      Class* cls;
      void* udata;
      
      // get the class that can serialize the item...
      items[i].forceClassInst( cls, udata );
      ++i; // prepare for going deep
      //... and traverse it.
      if ( self->m_owner->traverse( cls, udata ) ) 
      {
         //... then save the ID of the traversed item into our meta data.
         Private::ObjectData* subObj = traversing.back();
         objd->m_deps.push_back( subObj->m_id );
         traversing.pop_back();
      }
      else
      {
         // suspend our execution.
         return; 
      }
   }

   // we're done for now.
   traversing.pop_back();
   // was this the last element we should check?
   if( traversing.empty() )
   {
      // then declare the end of the object.
      self->m_owner->_p->setNextObject();
   }
   ctx->popCode();
}


void Storer::writeClassDict( DataWriter* wr )
{   
   // First, write the class dictionary.
   uint32 clsSize = (uint32) _p->m_clsVector.size();
   wr->write( clsSize );
   Private::ClassVector::const_iterator iter = _p->m_clsVector.begin();
   Private::ClassVector::const_iterator end = _p->m_clsVector.end();
   while( iter != end )
   {
      // for each class we must write its name and the module it is found in.
      Class* cls = *iter;
      Module* mod = cls->module();
      wr->write(cls->name());
      
      if ( mod != 0 )
      {
         wr->write( mod->name() );
         wr->write( mod->uri() );
      }
      else {
         wr->write("");
      }
         
      ++iter;
   }
}


bool Storer::doSerialize( DataWriter* wr )
{
   // write the object boundary count
   uint32 size = (uint32) _p->m_objBoundaries.size();
   wr->write( size );
   
   m_writer = wr;
   m_ctx->pushCode( &m_writeNext );
   m_writeNext.apply_( &m_writeNext, m_ctx );
   return _p == 0;
}

void Storer::WriteNext::apply_( const PStep* ps, VMContext* ctx )
{
   const WriteNext& self = *static_cast<const WriteNext*>(ps);
   
   // first, serialize the data of each item.
   register int32& pos = ctx->currentCode().m_seqId;
   int32 size = (int32) self.m_owner->_p->m_objBoundaries.size();
   DataWriter* wr = self.m_owner->m_writer;
   
   while( pos < size ) 
   {
      register uint32 i = pos;
      ++pos;
      if( ! self.m_owner->doSerializeItems( i, wr ) )
      {
         return;
      }
   }
     
   // we're done.
   delete self.m_owner->_p;
   self.m_owner->_p = 0;
   ctx->popCode();
}


bool Storer::doSerializeItems( uint32 pos, DataWriter* wr )
{
   // first, calculate the boundary of this item.
   uint32 start = pos == 0 ? 0 : _p->m_objBoundaries[pos-1];
   uint32 end = _p->m_objBoundaries[pos];
      
   // we must write at least 1 object data.
   fassert( end > start );
   
   // write a boundary to be sure about the synch
   uint32 boundary = 0xFECDBA98;
   wr->write( boundary );

   // write the item count
   uint32 size = end - start;
   wr->write( size );
   
   _p->m_partStart = start;
   _p->m_partEnd = end;
   
   m_ctx->pushCode( &m_writeNextPart );
   m_writeNextPart.apply( &m_writeNextPart, m_ctx );
   return _p->m_partEnd != _p->m_partStart;
}   


void Storer::WriteNextPart::apply_( const PStep* ps, VMContext* ctx )
{
   const WriteNextPart& self = *static_cast<const WriteNextPart*>(ps);
   Storer::Private* _p = self.m_owner->_p;
   DataWriter* wr = self.m_owner->m_writer;
   
   // we must record changes in _p->m_partStart because we may re-enter.
   uint32& start = _p->m_partStart;
   uint32& end = _p->m_partEnd;
   while( start < end )
   {
      // Get the item we must write.   
      const Private::ObjectData& obd = _p->m_objVector[start];
      // prepare for next loop
      ++start;

      // first, write the class ID referred by this object.
      uint32 clid = (uint32) obd.m_clsId;
      wr->write( clid );
      
      // then write the dep map.
      uint32 depSize = obd.m_deps.size();
      wr->write( depSize );
      
      for( uint32 depPos = 0; depPos < depSize; ++depPos )
      {
         uint32 depId = (uint32) obd.m_deps[depPos];
         wr->write( depId );
      }
      
      // And finally serialize the instance.
      Class* cls = _p->m_clsVector[clid];
            
      cls->store( 0, wr, obd.m_data );
      // went deep?
      if( ctx->currentCode().m_step != ps )
      {
         // suspend operations.
         return;
      }
      
   }
   
   // we're done -- changes in _p->m_partStart mark our completion.
   ctx->popCode();
}


}

/* end of storer.cpp */

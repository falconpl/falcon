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

#undef SRC
#define SRC "engine/storer.cpp"

#include <falcon/storer.h>
#include <falcon/class.h>
#include <falcon/datawriter.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>

#include <map>
#include <vector>
#include <list>
#include <set>

namespace Falcon {

class Storer::Private
{
public:   
   /** Map connecting each class with its own serialization ID. */
   typedef std::map<Class*, uint32> ClassMap;
   
   /** Map connecting each each object with its own serialization ID. */
   typedef std::map<const void*, uint32> ObjectMap;
   
   /** Vector of numeric IDs on which an object depends. */
   typedef std::vector<uint32> IDVector;
   
   /** Vector of items, used to store flat instances. */
   typedef std::list<Item> InstanceList;
   
   /** Map of objects that the user wants to store as a flat mantra. */
   typedef std::set<Mantra*> MantraSet;
   
   /** Data relative to a signle serialized item. */
   class ObjectData {
   public:
      const void* m_data;
      uint32 m_id;
      uint32 m_clsId;
      bool m_bIsGarbaged;
      
      /** Items to be given back while deserializing. */
      IDVector m_deps;
      
      /** Working array used during traversing.*/
      ItemArray m_theArray;
      
      ObjectData():
         m_data(0),
         m_id(0),
         m_clsId(0)
      {}
      
      ObjectData( const void* data, uint32 id,  uint32 cls, bool IsGarbaged ):
         m_data(data),
         m_id(id),
         m_clsId(cls),
         m_bIsGarbaged( IsGarbaged )
      {}
      
      ObjectData( const ObjectData& other ):
         m_data( other.m_data ),
         m_id( other.m_id ),
         m_clsId( other.m_clsId ),
         m_bIsGarbaged( other.m_bIsGarbaged )
      // ignore m_deps
      {}
   };
   
   typedef std::vector<ObjectData*> ObjectDataVector;
   typedef std::vector<ObjectData*> ObjectDataPtrVector;
   typedef std::vector<Class*> ClassVector;
   // Vector storing items of classes having flat data.
   InstanceList m_flatInstances;
   
   ClassMap m_classes;
   ClassVector m_clsVector;
   
   ObjectMap m_objects;
   ObjectDataVector m_objVector;
   // Objects as they are stored.
   IDVector m_storedVector;
   ObjectDataPtrVector m_objTraversing;
   
   MantraSet m_flatMantras;
   
   Private() {
   }
   
   ~Private() {
      ObjectDataVector::iterator obvi = m_objVector.begin();
      while( obvi != m_objVector.end() )
      {
         delete *obvi;
         ++obvi;
      }
   }
   
   void setNextObject( uint32 id )
   {
      m_storedVector.push_back( id );
   }
   
   
   ObjectData* addObject( Class* cls, const void* obj, bool& bIsNew, bool isGarbage = false )
   {
      // if the object is flat, it's never there.
      if( cls->isFlatInstance() )
      {
         const Item* data = static_cast<const Item*>(obj);
         m_flatInstances.push_back(*data);
         obj = &m_flatInstances.back();
      }
      else 
      {
         // It's not flat, we must search it.
         ObjectMap::const_iterator objIter = m_objects.find(obj);
         // if the object is alredy there...
         if( objIter != m_objects.end() )
         {
            //... we have already it all
            bIsNew = false;
            ObjectData* dt = m_objVector[objIter->second];
            return dt;
         }
      }
      
      // it's a new object -- manage it.
      bIsNew = true;
      uint32 objCount = (uint32) m_objVector.size();
      m_objects[obj] = objCount;
      
      // ... then, see if we need to save the class as well..
      uint32 clsId;
      ClassMap::const_iterator clsIter = m_classes.find(cls);
      if( clsIter != m_classes.end() )
      {
         // class was already there, get the ID
         clsId = clsIter->second;
      }
      else
      {
         // New class in town...
         clsId = m_classes.size();
         m_classes[cls] = clsId;
         m_clsVector.push_back( cls );
      }
      
      TRACE1("Adding object ID=%d(%p) with handler %s(%p) (CLSID=%d)%s",
               objCount, obj, cls->name().c_ize(), cls, clsId, (isGarbage ? " garbaged": "") );
      ObjectData* ndt = new ObjectData( obj, objCount, clsId, isGarbage );
      m_objVector.push_back( ndt );
      return ndt;
   }
   
   inline void addTraversing( ObjectData* obj )
   {
      m_objTraversing.push_back( obj );
   }
};

//===========================================================
//
//

Storer::Storer():
   _p(0),
   m_writer( new DataWriter(0) ),
   m_topData(0),
   m_topHandler(0),
   m_traverseNext( this ),
   m_writeNext(this)
{  
}


Storer::~Storer()
{
   if( !m_writer->isInGC() )
   {
      delete m_writer;
   }
   
   delete _p;
}

   
bool Storer::store( VMContext* ctx, Class* handler, const void* data, bool bInGarbage )
{
   if( _p == 0 )
   {
      _p = new Private;
   }
   
   m_topData = data;
   m_topHandler = handler;
   // First, create the Private that we use to unroll cycles.
   return traverse( ctx, handler, data, bInGarbage, true );
}

   
void Storer::addFlatMantra( Mantra* mantra )
{
   _p->m_flatMantras.insert(mantra);
}


bool Storer::isFlatMantra( const void* mantra )
{
   return _p->m_flatMantras.find( (Mantra*)mantra ) != _p->m_flatMantras.end();
}

void Storer::setStream( Stream* dataStream )
{
   m_writer->changeStream( dataStream, true );
}

bool Storer::commit( VMContext* ctx, Stream* dataStream )
{
   TRACE( "Storer::commit( %p, %p ) ", ctx, dataStream );
   try
   {
      if( dataStream != 0 )
      {
         m_writer->changeStream( dataStream, true );
      }
      writeClassTable( m_writer );
      writeInstanceTable( m_writer );

      // Serialize each item.
      return writeObjectTable( ctx, m_writer );
   }
   catch( ... )
   {
      delete _p;
      _p = 0;
      throw;
   }

   return true;
}


bool Storer::traverse( VMContext* ctx, Class* handler, const void* data, bool isGarbage, bool isTopLevel, void** obj )
{
   TRACE( "Entering traverse on handler %s ", handler->name().c_ize() );
   
   // first, save the item.
   bool bIsNew;
   Storer::Private::ObjectData* objd = _p->addObject( handler, data, bIsNew, isGarbage );
   if( obj != 0 )
   {
      *obj = objd;
   }
   
   if( isTopLevel )
   {
      _p->setNextObject( objd->m_id );
   }
   
   if( ! bIsNew )
   {
      //... was already there -- then we have nothing to do.
      return true;
   }
   
   // now that the object is saved, see if we must just store this as a mantra.
   if( isFlatMantra(data) )
   {
      // yep, no need to flatten
      return true;
   }
      
   // if the item is new, traverse its dependencies.
   ctx->pushCode( &m_traverseNext );
   int32 myDepth = ctx->codeDepth();
   handler->flatten( ctx, objd->m_theArray, (void*) data );
   if ( ctx->codeDepth() != myDepth )
   {
      // going deep? -- suspend processing and save current work object
      _p->addTraversing( objd );
      return false;
   }
   
   // nothing to traverse?
   if( objd->m_theArray.empty() )
   {
      ctx->popCode();
      return true;
   }
   
   // continue processing -- first saving current work object.
   _p->addTraversing( objd );
   // we know we're having as many deps as indicated in by the entity
   objd->m_deps.reserve(objd->m_theArray.length());
   // perform the sub-traversal.
   m_traverseNext.apply_( &m_traverseNext, ctx );
   // On completion, the PStep will pop itself, causing codeDepth < myDepth.
   return ctx->codeDepth() < myDepth;
}


void Storer::TraverseNext::apply_( const PStep* ps, VMContext* ctx )
{
   const TraverseNext* self = static_cast<const TraverseNext*>( ps );
   // get the object we're working on.
   Private::ObjectDataPtrVector& traversing = self->m_owner->_p->m_objTraversing;
   fassert( ! traversing.empty() );
   Private::ObjectData* objd = traversing.back();
   ItemArray& items = objd->m_theArray;
   
   int &i = ctx->currentCode().m_seqId;
   
   TRACE1( "TraverseNext::apply_ -- step %i", i );
   
   // The dependencies are now stored in items array.
   int length = (int) items.length();
   while( i < length ) 
   {
      Class* cls;
      void* udata;
      
      // get the class that can serialize the item...
      Item& current = items[i];
      current.forceClassInst( cls, udata );
   
      ++i; // prepare for going deep
      Private::ObjectData* newDep;
      bool bDidAll = self->m_owner->traverse( ctx, cls, udata,
               current.isUser() && current.isGarbage(), false, (void**) &newDep );
      // Always save the traversed item -- which is alwas added right now...
      objd->m_deps.push_back( newDep->m_id );
     
      if ( ! bDidAll ) 
      {
         // ... and suspend our execution in case the traversed item 
         // -- needs to be traversed again.
         return; 
      }
   }

   // This traversed item has been fully traversed.
   traversing.pop_back();
   ctx->popCode();
}


void Storer::writeClassTable( DataWriter* wr )
{   
   // First, write the class dictionary.
   uint32 clsSize = (uint32) _p->m_clsVector.size();
   TRACE( "Storer::writeClassTable writing %d classes ", clsSize );

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
         TRACE( "Storer::writeClassTable writing class %s from %s (%s)",
                     cls->name().c_ize(), mod->name().c_ize(), mod->uri().c_ize() );
         wr->write( mod->name() );
         wr->write( mod->uri() );
      }
      else {
         TRACE( "Storer::writeClassTable writing class %s without module", cls->name().c_ize() );
         wr->write("");
      }
         
      ++iter;
   }
}


void Storer::writeInstanceTable( DataWriter* wr )
{   
   // First, write the class dictionary.
   uint32 instSize = (uint32) _p->m_storedVector.size();
   TRACE( "Storer::writeInstanceTable writing %d instances ", instSize );
   wr->write( instSize );
   Private::IDVector::const_iterator iter = _p->m_storedVector.begin();
   Private::IDVector::const_iterator end = _p->m_storedVector.end();
   while( iter != end )
   {
      // for each class we must write its name and the module it is found in.
      uint32 itemID = *iter;
      wr->write(itemID);      
      ++iter;
   }
}


bool Storer::writeObjectTable( VMContext* ctx, DataWriter* wr )
{
   // write the object boundary count
   uint32 size = (uint32) _p->m_objVector.size();
   TRACE( "Storer::writeObjectTable writing %d objects", size );
   wr->write( size );

   ctx->pushCode( &m_writeNext );
   m_writeNext.apply_( &m_writeNext, ctx );
   // have we completely run without the need to call the VM?
   return _p == 0;
}

void Storer::WriteNext::apply_( const PStep* ps, VMContext* ctx )
{
   const WriteNext& self = *static_cast<const WriteNext*>(ps);
   
   // first, serialize the data of each item.
   register int32& pos = ctx->currentCode().m_seqId;
   int32 size = (int32) self.m_owner->_p->m_objVector.size();
   DataWriter* wr = self.m_owner->m_writer;
   TRACE( "Storer::WriteNext::apply_ writing object %d/%d", pos, size );
   
   while( pos < size ) 
   {
      register uint32 i = pos;
      ++pos;
      self.m_owner->writeObjectDeps( i, wr );
      if( ! self.m_owner->writeObject( ctx, i, wr ) )
      {
         return;
      }
   }
     
   // we're done.
   delete self.m_owner->_p;
   self.m_owner->_p = 0;
   self.m_owner->m_writer->flush();
   ctx->popCode();
}


void Storer::writeObjectDeps( uint32 pos, DataWriter* wr )
{
   // write a boundary to be sure about the synch
   uint32 boundary = 0xFECDBA98;
   wr->write( boundary );
      
   const Private::ObjectData& obd = *_p->m_objVector[pos];
   // first, write the class ID referred by this object.
   uint32 clid = (uint32) obd.m_clsId;
   wr->write( clid );
   
   // the object ID is not necessary, at it's sequential

   // then write the dep list.
   uint32 depSize = obd.m_deps.size();
   wr->write( depSize );

   for( uint32 depPos = 0; depPos < depSize; ++depPos )
   {
      uint32 depId = (uint32) obd.m_deps[depPos];
      wr->write( depId );
   }
}

bool Storer::writeObject( VMContext* ctx, uint32 pos, DataWriter* wr )
{   
   static Class* clsMantra = Engine::handlers()->mantraClass();
   
   // first, get the class that must serialize us.
   const Private::ObjectData& obd = *_p->m_objVector[pos];
   Class* cls = 0;
   // is this object a flattened mantra?
   if( isFlatMantra( obd.m_data ) ) {
      cls = clsMantra; // save only flat
   }
   else {
      uint32 clid = (uint32) obd.m_clsId;
      cls = _p->m_clsVector[clid];
   }
   
   const PStep* ps = ctx->currentCode().m_step;
   
   // then serialize us.
   wr->write( obd.m_bIsGarbaged );
   cls->store( ctx, wr, (void*) obd.m_data );
   // Return ture if we didn't go deep.
   return ctx->currentCode().m_step == ps;
}   


}

/* end of storer.cpp */

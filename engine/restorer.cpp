/*
   FALCON - The Falcon Programming Language.
   FILE: deserializer.cpp

   Helper for cyclic joint structure deserialization.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Oct 2011 16:06:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/restorer.cpp"

#include <falcon/restorer.h>
#include <falcon/string.h>
#include <falcon/class.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/module.h>
#include <falcon/modloader.h>
#include <falcon/modspace.h>
#include <falcon/symbol.h>
#include <falcon/engine.h>
#include <falcon/errors/unserializableerror.h>

#include <falcon/vmcontext.h>
#include <falcon/vm.h>

#include <vector>
#include <list>

namespace Falcon {

class Restorer::Private
{
public:   
   /** Vector of numeric IDs on which an object depends. */
   typedef std::vector<size_t> IDVector;
   
   /** Data relative to a signle serialized item. */
   class ObjectData {
   public:
      void* m_data;
      uint32 m_clsId;
      /** returned to the parent for the first time -- garbage it!*/
      bool m_bFirstTime;
      
      /** Items to be given back while deserializing. */
      IDVector m_deps;
      
      ObjectData():
         m_data(0),
         m_clsId(0),
         m_bFirstTime(true)
      {}
      
      ObjectData( void* data,  size_t cls ):
         m_data(data),
         m_clsId(cls),
         m_bFirstTime(true)
      {}
      
      ObjectData( const ObjectData& other ):
         m_data( other.m_data ),
         m_clsId( other.m_clsId ),
         m_bFirstTime(true)
      // ignore m_deps
      {}
   };
   
   /** Data relative to a signle serialized item. */
   class ClassInfo {
   public:
      String m_className;
      String m_moduleName;
      String m_moduleUri;
      
      Class* m_cls;
           
      ClassInfo():
         m_cls(0)
      {}
      
      ClassInfo( const ClassInfo& other ):
         m_className( other.m_className ),
         m_moduleName( other.m_moduleName ),
         m_moduleUri( other.m_moduleUri ),
         m_cls( other.m_cls )
      {}
   };
   
   typedef std::vector<ObjectData> ObjectDataVector;
   typedef std::vector<ClassInfo> ClassVector;
   typedef std::list<Item> ItemList;
   
   ClassVector m_clsVector;
   IDVector m_objList;
   ObjectDataVector m_objVector;
   
   ItemList m_flatItems;
   
   ItemArray m_flattener;
   
   /** Current ID in traversal */
   uint32 m_current;
   Private():
      m_current( 0 )
   {}
};

//===========================================================
//
//

Restorer::Restorer():
   _p(0),
   m_readNext( this ),
   m_unflattenNext( this ),
   m_stepLoadNextClass( this )
{
   m_reader = new DataReader;
}


Restorer::~Restorer()
{
   delete _p;
}


void Restorer::restore( VMContext* ctx, Stream* rd, ModSpace* space )
{
   delete _p;
   _p = new Private;
   
   try
   {
      m_reader->changeStream( rd, false, true );

      readClassTable();
   }
   catch( ... )
   {
      delete _p;
      _p = 0;
      throw;
   }

   loadClasses( ctx, space );
}


bool Restorer::next( Class*& handler, void*& data, bool &first )
{
   TRACE( "Restorer::next item %d", _p->m_current );
   
   if( _p != 0 && ( _p->m_current < _p->m_objList.size()) ) 
   {
      uint32 objID = _p->m_objList[_p->m_current];
      
      if( objID >= _p->m_objVector.size() )
      {
          throw new UnserializableError( ErrorParam(e_deser, __LINE__, SRC )
               .origin( ErrorParam::e_orig_runtime)
               .extra( String("Invalid instance ID at ").N(_p->m_current) ));
      }
      
      _p->m_current++;
      Private::ObjectData& objd = _p->m_objVector[objID];
      uint32 clsID = objd.m_clsId;
      handler = _p->m_clsVector[clsID].m_cls;
      data = objd.m_data;
      first = objd.m_bFirstTime;
      objd.m_bFirstTime = false;
      
      TRACE1( "Restorer::next -- restored %s %p", handler->name().c_ize(), data );
      return true;
   }
   
   return false;
}


bool Restorer::hasNext() const
{
   return _p != 0 && (_p->m_current < _p->m_objList.size());
}


uint32 Restorer::objCount() const
{
   return _p->m_objList.size();
}

//========================================================
// Private part
//

void Restorer::readClassTable()
{
   MESSAGE( "Restorer::readClassTable" );

   uint32 classCount;
   m_reader->read( classCount );
   TRACE1( "Restorer::readClassTable -- reading %d classes", classCount );
   _p->m_clsVector.resize( classCount );
   for (uint32 n = 0; n < classCount; n ++ )
   {
      Private::ClassInfo& info = _p->m_clsVector[n];
      m_reader->read( info.m_className );
      m_reader->read( info.m_moduleName );
      if( info.m_moduleName.size() > 0 )
      {
         m_reader->read( info.m_moduleUri );
      }
   }
   
}

void Restorer::loadClasses( VMContext* ctx, ModSpace* msp )
{
   MESSAGE( "Restorer::loadClasses" );

   ctx->pushData( Item( "ModSpace", msp ) );
   ctx->pushData( Item() );
   ctx->pushCode( &m_stepLoadNextClass );
}


void Restorer::readInstanceTable()
{
   MESSAGE( "Restorer::readInstanceTable" );
   uint32 instCount;
   m_reader->read( instCount );
   TRACE1( "Restorer::readInstanceTable -- reading %d instances", instCount );
   
   _p->m_objList.resize( instCount );   
   for (uint32 n = 0; n < instCount; n ++ )
   {      
      uint32 objID;
      m_reader->read( objID );
      _p->m_objList[n] = objID;
   }
}


bool Restorer::readObjectTable( VMContext* ctx )
{
   MESSAGE( "Restorer::readObjectTable" );
   uint32 instCount;
   m_reader->read( instCount );
   TRACE1( "Restorer::readObjectTable -- reading %d objects", instCount );
   
   _p->m_objVector.resize( instCount );
   
   int32 pcount = ctx->codeDepth();
   ctx->pushCode( &m_readNext );
   m_readNext.apply_( &m_readNext, ctx );
   
   // unflatten will be called by m_readNext at proper time.
   // have we completely run without the need to call the VM?
   return pcount == ctx->codeDepth();
}


bool Restorer::unflatten( VMContext* ctx )
{
   MESSAGE( "Restorer::unflatten" );
   int32 pcount = ctx->codeDepth();
   ctx->pushCode( &m_unflattenNext );
   // if executing up to the end, unflatten_next will destroy P
   m_unflattenNext.apply_( &m_unflattenNext, ctx );
   return pcount == ctx->codeDepth();
}

//========================================================
// VM Steps
//


void Restorer::ReadNext::apply_( const PStep* ps, VMContext* ctx )
{
   uint32 depsCount, marker;
   
   const Restorer::ReadNext* self = static_cast<const Restorer::ReadNext*>(ps);
   DataReader* m_reader = self->m_owner->m_reader;
   Restorer::Private* _p = self->m_owner->_p;
   Restorer::Private::ObjectDataVector& objects = _p->m_objVector;
   
   int32& seq = ctx->currentCode().m_seqId;
   uint32 objCount = objects.size();
   TRACE( "Restorer::ReadNext::apply_ with sequence %d/%d", seq, objCount );
   
   if( seq > 0 )
   {
      if( _p->m_clsVector[objects[seq-1].m_clsId].m_cls->isFlatInstance() )
      {
         // classes with flat instances need the restorer to prepare the objects.
         _p->m_flatItems.push_back( ctx->topData() );
         objects[seq-1].m_data = &_p->m_flatItems.back();
      }
      else {
         objects[seq-1].m_data = ctx->topData().asInst();
      }
      ctx->popData();
   }

   while( ((uint32)seq) < objCount )
   {
      TRACE1( "Restorer::ReadNext::apply_ -- next object %d/%d", seq, objCount );
      // get the stored object.
      Restorer::Private::ObjectData& objd = objects[seq];

      // prepare for next step
      seq = ((uint32)seq) + 1;

      m_reader->read( marker );
      if( marker != 0xFECDBA98)
      {
         throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra( String("Sync marker at object ").N(seq-1)));
      }
      
      m_reader->read( objd.m_clsId );
      if( objd.m_clsId >= _p->m_clsVector.size() )
      {
         throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra( String("Class ID at object ").N(seq-1)));
      }
      
      m_reader->read( depsCount );
      TRACE2( "Restorer::ReadNext::apply_ -- restoring object %d for class %d (%s) depsCount %d ", 
            seq, objd.m_clsId, _p->m_clsVector[objd.m_clsId].m_className.c_ize(), depsCount);
      if( depsCount > 0 )
      {
         objd.m_deps.resize( depsCount );
         for( uint32 i = 0; i < depsCount; ++ i )
         {
            uint32 objID;
            m_reader->read( objID );
            objd.m_deps[i] = objID;
            // the objects have been already resized.
            if( objID >= objects.size() )
            {
               throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
                  .origin( ErrorParam::e_orig_runtime )
                  .extra( String("Dependency ID ").N(i).A(" at object ").N(seq-1)));
            }
         }
      }
      
      // ok, the object is loaded. Now we got to deserialize the data.
      Class* cls = _p->m_clsVector[ objd.m_clsId ].m_cls;
      cls->restore( ctx, m_reader );
      if( ctx->currentCode().m_step != self )
      {
         // we went deep
         MESSAGE1( "Restorer::ReadNext::apply_ -- went deep." );
         return;
      }
      else {
         // otherwise the restore was atomic and we can go on.
         if( cls->isFlatInstance() )
         {
            // classes with flat instances need the restorer to prepare the objects.
            _p->m_flatItems.push_back( ctx->topData() );
            objd.m_data = &_p->m_flatItems.back();
         }
         else {
            objd.m_data = ctx->topData().asInst();
         }
         ctx->popData();
      }
   }
   
   // completed -- it's time to perform unflattening
   ctx->popCode();
   self->m_owner->unflatten( ctx );
}


void Restorer::UnflattenNext::apply_( const PStep* ps, VMContext* ctx )
{
   const Restorer::UnflattenNext* self = static_cast<const Restorer::UnflattenNext*>(ps);
   Restorer::Private* _p = self->m_owner->_p;
   Restorer::Private::ObjectDataVector& objects = _p->m_objVector;
   
   int32& seq = ctx->currentCode().m_seqId;
   uint32 objCount = objects.size();
   TRACE( "Restorer::UnflattenNext::apply_ with sequence %d/%d", seq, objCount );
   
   while( ((uint32)seq) < objCount )
   {
      Restorer::Private::ObjectData& objd = objects[seq];
      // prepare for next step
      ++seq;
      
      // have we dependencies?
      if( objd.m_deps.size() > 0 )
      {
         
         // Load them in the temporary array
         _p->m_flattener.resize( objd.m_deps.size() );
         for( uint32 i = 0; i < objd.m_deps.size(); ++i )
         {
            Private::ObjectData& tgtdata = objects[objd.m_deps[i]];
            Class* cls = _p->m_clsVector[ tgtdata.m_clsId ].m_cls;
            if( cls->isFlatInstance() )
            {
               _p->m_flattener[i] = *static_cast<Item*>(tgtdata.m_data);
            }
            else
            {
               _p->m_flattener[i].setUser( cls, tgtdata.m_data );
            }
         }
         
         // ask the class to use the objects.
         Class* cls = _p->m_clsVector[ objd.m_clsId ].m_cls;
         cls->unflatten( ctx, _p->m_flattener, objd.m_data );
         if( ctx->currentCode().m_step != self )
         {
            // we went deep
            return;
         }
      }
   }
   
   // we're done here.
   ctx->popCode();   
}
   

void Restorer::PStepLoadNextClass::apply_( const PStep* ps, VMContext* ctx )
{
   MESSAGE( "Restorer::PStepLoadNextClass::apply_" );
   
   const Restorer::PStepLoadNextClass* self = static_cast<const Restorer::PStepLoadNextClass*>(ps);
   Restorer* restorer = self->m_owner;
   ModSpace* ms = static_cast<ModSpace*>(ctx->opcodeParam(1).asOpaque());
   int& seqId = ctx->currentCode().m_seqId;
   int size = (int) restorer->_p->m_clsVector.size();

   if (seqId > 0 )
   {
      Mantra* mantra = static_cast<Mantra*>(ctx->topData().asInst());
      Private::ClassInfo& cinfo = restorer->_p->m_clsVector[seqId-1];
      if( mantra->isCompatibleWith( Mantra::e_c_class ) )
      {
         cinfo.m_cls = static_cast<Class*>(mantra);
      }
   }
   ctx->popData();

   int depth = ctx->codeDepth();
   while( seqId < size )
   {
      TRACE1( "Restorer::PStepLoadNextClass::apply_ %d/%d", seqId, size );

      Private::ClassInfo& cinfo = restorer->_p->m_clsVector[seqId++];
      ms->findDynamicMantra(
               ctx, cinfo.m_moduleUri, cinfo.m_moduleName, cinfo.m_className );
      if( ctx->codeDepth() != depth  ) {
         return;
      }

      Mantra* mantra = static_cast<Mantra*>(ctx->topData().asInst());
      if( mantra->isCompatibleWith( Mantra::e_c_class ) )
      {
         cinfo.m_cls = static_cast<Class*>(mantra);
      }
      // remove stored mantra
      ctx->popData();
   }

   ctx->popData();
   ctx->popCode();

   restorer->readInstanceTable();
   restorer->readObjectTable( ctx );
}
}

/* end of deserializer.cpp */

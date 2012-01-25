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
      size_t m_clsId;
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
   
   DataReader m_reader;
   
   /** Current ID in traversal */
   uint32 m_current;
   Private( Stream* str ):
      m_reader( str ),
      m_current( 0 )
   {}
};

//===========================================================
//
//

Restorer::Restorer( VMContext* ctx ):
   _p(0),
   m_ctx( ctx ),
   m_readNext( this ),
   m_unflattenNext( this ),
   m_linkNext( this )
{       
}


Restorer::~Restorer()
{
   delete _p;
}


bool Restorer::restore( Stream* rd, ModSpace* space, ModLoader* ml )
{
   delete _p;
   _p = new Private(rd );
   
   try
   {
      readClassTable();  
      if( ! loadClasses( space, ml ) )
      {
         return false;
      }
      
      readInstanceTable();
   
      return readObjectTable();   
   }
   catch( ... )
   {
      delete _p;
      _p = 0;
      throw;
   }
}


bool Restorer::next( Class*& handler, void*& data, bool &first )
{
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
   uint32 classCount;
   _p->m_reader.read( classCount );
   _p->m_clsVector.resize( classCount );
   for (uint32 n = 0; n < classCount; n ++ )
   {
      Private::ClassInfo& info = _p->m_clsVector[n];
      _p->m_reader.read( info.m_className );
      _p->m_reader.read( info.m_moduleName );
      if( info.m_moduleName.size() > 0 )
      {
         _p->m_reader.read( info.m_moduleUri );
      }
   }
   
}

bool Restorer::loadClasses( ModSpace* msp, ModLoader* ml )
{
   bool addedMod = false;
   
   Private::ClassVector::iterator iter = _p->m_clsVector.begin();
   while( _p->m_clsVector.end() != iter )
   {
      Private::ClassInfo& cinfo = *iter;
      cinfo.m_cls = msp->findDynamicClass( 
            ml, cinfo.m_moduleUri, cinfo.m_moduleName, cinfo.m_className, 
            addedMod );
      
      ++iter;
   }
   
   if( addedMod )
   {
      // If there aren't new modules, these will be no-ops.
      Error* errs = msp->link();
      if( errs != 0 ) throw errs;

      // prepare a step to be on the bright side in case of linking
      m_ctx->pushCode( &m_linkNext );
      // see if the loaded modules (if any) need a startup.
      if( msp->readyVM( m_ctx ) )
      {
         // we must return the control to the caller as the modules require 
         // initialization.
         return false;
      }
      // ok, we didn't need to link anything.
      m_ctx->popCode();
   }
   
   return true;
}


void Restorer::readInstanceTable()
{
   uint32 instCount;
   _p->m_reader.read( instCount );
   _p->m_objList.resize( instCount );
   for (uint32 n = 0; n < instCount; n ++ )
   {      
      uint32 objID;
      _p->m_reader.read( objID );
      _p->m_objList[n] = objID;
   }
}


bool Restorer::readObjectTable()
{
   uint32 instCount;
   _p->m_reader.read( instCount );
   _p->m_objVector.resize( instCount );
   
   int32 pcount = m_ctx->codeDepth();
   m_ctx->pushCode( &m_readNext );
   m_readNext.apply_( &m_readNext, m_ctx );
   
   // unflatten will be called by m_readNext at proper time.
   // have we completely run without the need to call the VM?
   return pcount == m_ctx->codeDepth();
}


bool Restorer::unflatten()
{
   int32 pcount = m_ctx->codeDepth();
   m_ctx->pushCode( &m_unflattenNext );
   // if executing up to the end, unflatten_next will destroy P
   m_unflattenNext.apply_( &m_unflattenNext, m_ctx );
   return pcount == m_ctx->codeDepth();
}

//========================================================
// VM Steps
//


void Restorer::ReadNext::apply_( const PStep* ps, VMContext* ctx )
{
   uint32 depsCount, marker;
   
   const Restorer::ReadNext* self = static_cast<const Restorer::ReadNext*>(ps);
   Restorer::Private* _p = self->m_owner->_p;
   Restorer::Private::ObjectDataVector& objects = _p->m_objVector;
   
   int32& seq = ctx->currentCode().m_seqId;
   while( ((uint32)seq) < objects.size() )
   {
      Restorer::Private::ObjectData& objd = objects[seq];
      // prepare for next step
      ++seq;
      //TODO: check well known classes
      _p->m_reader.read( marker );
      if( marker != 0xFECDBA98)
      {
         throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra( String("Sync marker at object ").N(seq-1)));
      }
      
      _p->m_reader.read( objd.m_clsId );
      //TODO: check well known classes
      if( objd.m_clsId >= _p->m_clsVector.size() )
      {
         throw new UnserializableError( ErrorParam( e_deser, __LINE__, SRC )
            .origin( ErrorParam::e_orig_runtime )
            .extra( String("Class ID at object ").N(seq-1)));
      }
      
      _p->m_reader.read( depsCount );
      if( depsCount > 0 )
      {
         objd.m_deps.resize( depsCount );
         for( uint32 i = 0; i < depsCount; ++ i )
         {
            uint32 objID;
            _p->m_reader.read( objID );
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
      if( cls->isFlatInstance() )
      {
         // classes with flat instances need the restorer to prepare the objects.
         _p->m_flatItems.push_back( Item() );
         objd.m_data = &_p->m_flatItems.back();
      }
      
      cls->restore( ctx, &_p->m_reader, objd.m_data );
      if( ctx->currentCode().m_step != self )
      {
         // we went deep
         return;
      }
      
      // otherwise the restore was atomic and we can go on.
   }
   
   // completed -- it's time to perform unflattening
   ctx->popCode();
   self->m_owner->unflatten();   
}


void Restorer::UnflattenNext::apply_( const PStep* ps, VMContext* ctx )
{
   const Restorer::UnflattenNext* self = static_cast<const Restorer::UnflattenNext*>(ps);
   Restorer::Private* _p = self->m_owner->_p;
   Restorer::Private::ObjectDataVector& objects = _p->m_objVector;
   
   int32& seq = ctx->currentCode().m_seqId;
   while( ((uint32)seq) < objects.size() )
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
   

void Restorer::LinkNext::apply_( const PStep* ps, VMContext* ctx )
{
   const Restorer::LinkNext* self = static_cast<const Restorer::LinkNext*>(ps);
   // we're not interested in being called again.
   ctx->popCode();
   // just ask for the other steps to be completed.
   try {
      self->m_owner->readInstanceTable();   
      self->m_owner->readObjectTable();  
   }
   catch( ... )
   {
      delete self->m_owner->_p;
      self->m_owner->_p = 0;
      throw;
   }
}
}

/* end of deserializer.cpp */

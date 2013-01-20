/*
   FALCON - The Falcon Programming Language
   FILE: engine.cpp

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 12:39:16 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#undef SRC
#define SRC "engine/engine.cpp"

#include <map>
#include <stdio.h>

#include <falcon/trace.h>
#include <falcon/engine.h>

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/string.h>
#include <falcon/mt.h>
#include <falcon/pool.h>
#include <falcon/log.h>

#include <falcon/errors/codeerror.h>


//--- Virtual file systems ---
#include <falcon/vfs_file.h>

//--- standard transcoder headers ---

#include <falcon/tc/transcoderc.h>
#include <falcon/tc/transcoderutf8.h>

//--- core function headers ---
#include <falcon/cm/coremodule.h>
#include <falcon/builtin/compare.h>
#include <falcon/builtin/derivedfrom.h>
#include <falcon/builtin/len.h>
#include <falcon/builtin/minmax.h>
#include <falcon/builtin/typeid.h>
#include <falcon/builtin/clone.h>
#include <falcon/builtin/classname.h>
#include <falcon/builtin/baseclass.h>
#include <falcon/builtin/describe.h>
#include <falcon/builtin/tostring.h>

#include <falcon/bom.h>
#include <falcon/stdsteps.h>
#include <falcon/synfunc.h>

//--- object headers ---
#include <falcon/pseudofunc.h>
#include <falcon/collector.h>

//--- type headers ---
#include <falcon/classes/classfunction.h>
#include <falcon/classes/classsynfunc.h>
#include <falcon/classes/classnil.h>
#include <falcon/classes/classbool.h>
#include <falcon/classes/classcloseddata.h>
#include <falcon/classes/classclosure.h>
#include <falcon/classes/classcomposition.h>
#include <falcon/classes/classint.h>
#include <falcon/classes/classnumeric.h>
#include <falcon/classes/classstring.h>
#include <falcon/classes/classrange.h>
#include <falcon/classes/classarray.h>
#include <falcon/classes/classdict.h>
#include <falcon/classes/classgeneric.h>
#include <falcon/classes/classnumeric.h>
#include <falcon/classes/classmantra.h>
#include <falcon/classes/classmethod.h>
#include <falcon/classes/classshared.h>
#include <falcon/classes/metaclass.h>
#include <falcon/classes/metafalconclass.h>
#include <falcon/classes/metahyperclass.h>
#include <falcon/classes/classstorer.h>
#include <falcon/classes/classrestorer.h>
#include <falcon/classes/classstream.h>

#include <falcon/classes/classmodule.h>

#include <falcon/classes/classtreestep.h>
#include <falcon/classes/classstatement.h>
#include <falcon/classes/classexpression.h>
#include <falcon/classes/classsyntree.h>
#include <falcon/classes/classsymbol.h>
#include <falcon/classes/classrawmem.h>
#include <falcon/synclasses.h>

#include <falcon/psteps/exprinherit.h>
#include <falcon/psteps/stmtselect.h>

#include <falcon/prototypeclass.h>

#include <falcon/stderrors.h>
#include <falcon/modspace.h>
#include <falcon/symbol.h>

#include <falcon/item.h>         // for builtin

#include <falcon/paranoid.h>
#include <map>
#include <deque>

namespace Falcon
{


//=======================================================
// Private classes known by the engine -- Utils
//

class TranscoderMap: public std::map<String, Transcoder*>
{
};

class PredefMap: public std::map<String, Item>
{
};

class MantraMap: public std::map<String, Mantra*>
{
};

class PoolList: public std::deque<Pool* >
{
};


class SymbolPool
{
public:

   inline Symbol* get(const String& name, int poolId ) {
      bool isFirst = false;
      return get( name, poolId, isFirst );
   }

   inline Symbol* get(const String& name, int poolId, bool& isFirst )
   {
      Symbol *s;
      m_mtx[poolId].lock();
      SymbolSet::iterator iter = m_symbols[poolId].find(&name);
      if( iter == m_symbols[poolId].end() ) {
         s = new Symbol( name, poolId == 1 );
         m_symbols[poolId][&s->name()] = s;
         isFirst = true;
      }
      else {
         s = iter->second;
         s->m_counter++;
         isFirst = false;
      }
      m_mtx[poolId].unlock();

      return s;
   }

   inline void release( Symbol* s, int poolId ) {
     m_mtx[poolId].lock();
     if( --s->m_counter == 0 ) {
        m_symbols[poolId].erase(&s->name());
        delete s;
     }
     m_mtx[poolId].unlock();
   }

   inline void ref( Symbol* s, int poolId ) {
     m_mtx[poolId].lock();
     s->m_counter++;
     m_mtx[poolId].unlock();
   }

   ~ SymbolPool() {
      for( int i = 0; i < 2; ++i ) {
         SymbolSet::iterator iter = m_symbols[i].begin();
         SymbolSet::iterator end = m_symbols[i].end();

         while( iter != end ) {
            delete iter->second;
            ++iter;
         }
      }
   }

private:
   class StringPtrCheck {
   public:
      inline bool operator ()( const String* s1, const String *s2 ) {
         return *s1 < *s2;
      }
   };

   typedef std::map<const String*, Symbol*, StringPtrCheck> SymbolSet;
   SymbolSet m_symbols[2];
   Mutex m_mtx[2];
};

//=======================================================
// Engine static declarations
//

Engine* Engine::m_instance = 0;

//=======================================================
// Engine implementation
//

Engine::Engine()
{
   MESSAGE( "Engine creation started" );
   #ifdef FALCON_SYSTEM_WIN
   m_bWindowsNamesConversion = true;
   #else
   m_bWindowsNamesConversion = false;
   #endif

   if( m_instance != 0 )
   {
      throw new CodeError( ErrorParam(e_paranoid, __LINE__, SRC)
         .extra( "Engine double initialization"));
   }   
   m_instance = this; // modules need the engine.

   m_mtx = new Mutex;
   m_log = new Log;
   m_collector = new Collector;   

   //=====================================
   // Standard file systems.
   //
   m_vfs.addVFS("", new VFSFile );
   m_vfs.addVFS("file", new VFSFile );


   //=====================================
   // Initialization of standard deep types.
   //
   m_symbols = new SymbolPool;
   m_baseSymbol = m_symbols->get("$base",0);
   m_functionClass = new ClassFunction;
   m_stringClass = new ClassString;
   m_rangeClass = new ClassRange;
   m_arrayClass = new ClassArray;
   m_dictClass = new ClassDict;
   m_protoClass = new PrototypeClass;
   m_metaClass = new MetaClass;
   m_metaFalconClass = new MetaFalconClass;
   m_metaHyperClass = new MetaHyperClass;
   m_mantraClass = new ClassMantra;
   m_synFuncClass = new ClassSynFunc;
   m_genericClass = new ClassGeneric;
   m_sharedClass = new ClassShared;
   
   // Notice: rawMem is not reflected, is used only in extensions.
   m_rawMemClass = new ClassRawMem();
   
   // Initialization of the class vector.
   m_classes[FLC_ITEM_NIL] = new ClassNil;
   m_classes[FLC_ITEM_BOOL] = new ClassBool;
   m_classes[FLC_ITEM_INT] = new ClassInt;
   m_classes[FLC_ITEM_NUM] = new ClassNumeric;
   m_classes[FLC_ITEM_METHOD] = new ClassMethod;
   
   m_bom = new BOM;
   
   //===========================================
   // Subsystems initialization
   //   
   Item::init( this );
   
   m_pools = new PoolList;

   //=====================================
   // Adding standard transcoders.
   //

   m_tcoders = new TranscoderMap;
   addTranscoder( new TranscoderC );
   addTranscoder( new TranscoderUTF8 );

   //============================================
   // Creating predefined symbols
   //
   m_predefs = new PredefMap;
   m_stdSteps = new StdSteps;
   m_stdErrors = new StdErrors; 

   //=====================================
   // Adding standard pseudo functions.
   //

   m_mantras = new MantraMap;
   addMantra(new Ext::Compare);
   addMantra(new Ext::DerivedFrom);
   addMantra(new Ext::Len);
   addMantra(new Ext::Max);
   addMantra(new Ext::Min);
   addMantra(new Ext::TypeId);
   addMantra(new Ext::Clone);
   addMantra(new Ext::ClassName);
   addMantra(new Ext::BaseClass);
   addMantra(new Ext::Describe);
   addMantra(new Ext::ToString);

   //============================================
   // Creating singletons
   //
      
   addMantra( m_functionClass );
   addMantra( m_stringClass );
   addMantra( m_arrayClass );
   addMantra( m_dictClass );
   addMantra( m_protoClass );
   addMantra( m_metaClass );
   addMantra( m_metaFalconClass );
   addMantra( m_metaHyperClass );
   addMantra( m_mantraClass ); 
   addMantra( m_synFuncClass );
   addMantra( m_genericClass );
   addMantra( m_rangeClass  );
   addMantra( m_sharedClass );

   addMantra( m_classes[FLC_ITEM_NIL] );
   addMantra( m_classes[FLC_ITEM_BOOL] );
   addMantra( m_classes[FLC_ITEM_INT] );
   addMantra( m_classes[FLC_ITEM_NUM] );
   addMantra( m_classes[FLC_ITEM_METHOD] );
   
   addBuiltin( "NilType", (int64) FLC_ITEM_NIL );
   addBuiltin( "BoolType", (int64) FLC_ITEM_BOOL );
   addBuiltin( "IntType", (int64) FLC_ITEM_INT );
   addBuiltin( "NumericType", (int64) FLC_ITEM_INT ); // same as int
   addBuiltin( "MethodType", (int64) FLC_ITEM_METHOD );
   
   addBuiltin( "ObjectType", (int64) FLC_ITEM_USER );
   addBuiltin( "FunctionType", (int64) FLC_CLASS_ID_FUNC );
   addBuiltin( "StringType", (int64) FLC_CLASS_ID_STRING );
   addBuiltin( "ArrayType", (int64) FLC_CLASS_ID_ARRAY );
   addBuiltin( "DictType", (int64) FLC_CLASS_ID_DICT );
   addBuiltin( "RangeType", (int64) FLC_CLASS_ID_RANGE );
   addBuiltin( "ClassType", (int64) FLC_CLASS_ID_CLASS );
   addBuiltin( "ProtoType", (int64) FLC_CLASS_ID_PROTO );
      
   m_stdErrors->addBuiltins();
   //=====================================
   // Syntax Reflection
   //
      
   m_closureClass = new ClassClosure;
   m_closedDataClass = new ClassClosedData;
   m_restorerClass = new ClassRestorer;
   m_storerClass = new ClassStorer;
   m_streamClass = new ClassStream;

   m_symbolClass = new ClassSymbol;
   
   ClassTreeStep* ctreeStep = new ClassTreeStep;
   m_treeStepClass = ctreeStep;
   m_statementClass = new ClassStatement(ctreeStep);
   m_exprClass = new ClassExpression(ctreeStep);
   m_syntreeClass = new ClassSynTree(ctreeStep, static_cast<ClassSymbol*>(m_symbolClass));

   m_moduleClass = new ClassModule;

   addMantra(m_closureClass);
   addMantra(m_treeStepClass);
   addMantra(m_statementClass);
   addMantra(m_exprClass);
   addMantra(m_syntreeClass);
   addMantra(m_symbolClass); 
   addMantra(m_moduleClass);
   addMantra(m_restorerClass);
   addMantra(m_storerClass);
   addMantra(m_streamClass);

   addMantra( new ClassComposition );
   
   ExprInherit::IRequirement::registerMantra();
   SelectRequirement::registerMantra();
   
   m_synClasses = new SynClasses(m_syntreeClass, m_statementClass, m_exprClass );
   m_synClasses->subscribe( this );
   
   //=====================================
   // The Core Module
   //
   m_core  = new CoreModule;
      
   MESSAGE( "Engine creation complete" );
}


Engine::~Engine()
{
   MESSAGE( "Engine destruction started" );

   m_collector->stop();
   
   /** Bye bye core... */
   m_core->decref();

   // ===============================
   // DO NOT Delete standard item classes -- they are mantras
   //
   delete m_symbols;
   
   // ===============================
   // delete registered transcoders
   //

   {
      TranscoderMap::iterator iter = m_tcoders->begin();
      while( iter != m_tcoders->end() )
      {
         delete iter->second;
         ++iter;
      }
   }
   
   delete m_tcoders;
   
    // ===============================
   // delete mantras transcoders
   //

   {
      MantraMap::iterator iter = m_mantras->begin();
      while( iter != m_mantras->end() )
      {
         delete iter->second;
         ++iter;
      }
   }
   
   delete m_mantras;
   
   // ===============================
   // delete builtin symbols
   //   
   delete m_predefs;   
   delete m_synClasses;
   delete m_closedDataClass;
   
   //============================================
   // Delete singletons
   //
   delete m_bom;
   delete m_stdSteps;
   delete m_stdErrors;

   delete m_collector;
   delete m_mtx;
   
   //============================================
   // Delete pools
   //
   {
      PoolList::iterator iter = m_pools->begin();
      while( iter != m_pools->end() )
      {
         delete *iter;
         ++iter;
      }
      
      delete m_pools;
   }

   delete m_log;
   MESSAGE( "Engine destroyed" );
}

void Engine::init()
{
   MESSAGE( "Engine init()" );
   if( m_instance == 0 )
   {
      m_instance = new Engine;

      // TODO
      // m_instance->collector()->start();
   }
}

void Engine::shutdown()
{
   MESSAGE( "Engine shutdown started" );
   fassert( m_instance != 0 );
   if( m_instance != 0 )
   {
      // TODO
      // m_instance->collector()->start();

      delete m_instance;
      m_instance = 0;
      MESSAGE( "Engine shutdown complete" );
   }
}


void Engine::die( const String& msg )
{
   String res = msg;
   fprintf( stderr, "%s\n", msg.c_ize() );
   exit(1);
}

//=====================================================
// Global settings
//

bool Engine::isWindows() const
{
   fassert( m_instance != 0 );
   return m_instance->m_bWindowsNamesConversion;
}


//=====================================================
// Transcoding
//

bool Engine::addTranscoder( Transcoder* ts )
{
   m_mtx->lock();
   TranscoderMap::iterator iter = m_tcoders->find(ts->name());
   if ( iter != m_tcoders->end() )
   {
      m_mtx->unlock();
      return false;
   }

   (*m_tcoders)[ts->name()] = ts;
   m_mtx->unlock();
   return true;
}


Transcoder* Engine::getTranscoder( const String& name )
{
   m_mtx->lock();
   TranscoderMap::iterator iter = m_tcoders->find(name);
   if ( iter == m_tcoders->end() )
   {
      m_mtx->unlock();
      return 0;
   }

   Transcoder* ret = iter->second;
   m_mtx->unlock();
   return ret;
}

//=====================================================
// Pseudofunctions
//

bool Engine::addMantra( Mantra* pf )
{
   m_mtx->lock();
   MantraMap::iterator iter = m_mantras->find(pf->name());
   if ( iter != m_mantras->end() )
   {
      m_mtx->unlock();
      return false;
   }

   (*m_mantras)[pf->name()] = pf;
   // force to add the builtin, even if we had it.
   // mantras have precedence.
   (*m_predefs)[ pf->name() ] = Item(pf->handler(), pf);   
   
   m_mtx->unlock();
   
   return true;
}

Mantra* Engine::getMantra( const String& name, Mantra::t_category cat ) const
{
   // first, see if we have something with that name in the mantras
   m_mtx->lock();
   MantraMap::iterator iter = m_mantras->find(name);
   if ( iter == m_mantras->end() )
   {
      m_mtx->unlock();
      return 0;
   }

   Mantra* ret = iter->second;
   m_mtx->unlock();
   
   // then check if the item is what we thought it should be.
   if( ret->isCompatibleWith( cat ) )
   {
      return ret;
   }
   
   
   // We found it, but it wasn't what we were searching for.
   return 0;
}


bool Engine::addBuiltin( const String& name, const Item& value )
{
   m_mtx->lock();
   PredefMap::iterator pos = m_predefs->find( name );
   if( pos != m_predefs->end() )
   {
      m_mtx->unlock();
      return false;
   }
   
   (*m_predefs)[ name ] = value;
   m_mtx->unlock();
   return true;
}

const Item* Engine::getBuiltin( const String& name ) const
{
   m_mtx->lock();
   PredefMap::iterator pos = m_predefs->find( name );
   if( pos != m_predefs->end() )
   {
      const Item* item = &pos->second;
      m_mtx->unlock();
      return item;
   }
   
   m_mtx->unlock();
   return 0;
}

void Engine::addPool( Pool* p ) 
{
   m_mtx->lock();
   m_pools->push_back( p );
   m_mtx->unlock();
}


//=====================================================
// Global objects
//

Engine* Engine::instance()
{
   fassert( m_instance != 0 );
   return m_instance;
}
 
Collector* Engine::collector()
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   static Collector* coll = m_instance->m_collector;
   return coll;
}

GCToken* Engine::GC_store( const Class* cls, void* data )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   return m_instance->m_collector->store( cls, data );
}

GCLock* Engine::GC_storeLocked( const Class* cls, void* data )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   return m_instance->m_collector->storeLocked( cls, data );
}

GCToken* Engine::GC_H_store( const Class* cls, void* data, const String& file, int line )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   return m_instance->m_collector->H_store( cls, data, file, line );
}

GCLock* Engine::GC_H_storeLocked( const Class* cls, void* data, const String& src, int line )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   return m_instance->m_collector->H_storeLocked( cls, data, src, line );
}

GCLock* Engine::GC_lock( const Item& item )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   return m_instance->m_collector->lock( item );
}

void Engine::GC_unlock( GCLock* lock )
{
   fassert( m_instance != 0 );
   fassert( m_instance->m_collector != 0 );
   m_instance->m_collector->unlock( lock );
}

Log* Engine::log() const
{
   fassert( m_instance != 0 );
   return m_instance->m_log;
}
   
//=====================================================
// Type handlers
//

Class* Engine::getTypeClass( int type )
{
   PARANOID("type out of range", (type < FLC_ITEM_COUNT) );
   return m_classes[type];
}

Class* Engine::functionClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_functionClass;
}


Class* Engine::stringClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_stringClass;
}

Class* Engine::rangeClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_rangeClass;
}


Class* Engine::arrayClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_arrayClass;
}

Class* Engine::dictClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_dictClass;
}

Class* Engine::protoClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_protoClass;
}

Class* Engine::metaClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_metaClass;
}

Class* Engine::metaFalconClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_metaFalconClass;
}

Class* Engine::metaHyperClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_metaHyperClass;
}

Class* Engine::mantraClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_mantraClass;
}

Class* Engine::synFuncClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_synFuncClass;
}

Class* Engine::genericClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_genericClass;
}

Class* Engine::treeStepClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_treeStepClass;
}

Class* Engine::statementClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_statementClass;
}

Class* Engine::expressionClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_exprClass;
}

Class* Engine::syntreeClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_syntreeClass;
}

Class* Engine::symbolClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_symbolClass;
}

Class* Engine::closureClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_closureClass;
}


Class* Engine::closedDataClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_closedDataClass;
}


Class* Engine::storerClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_storerClass;
}

Class* Engine::restorerClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_restorerClass;
}

Class* Engine::streamClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_streamClass;
}

ClassRawMem* Engine::rawMemClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_rawMemClass;
}

ClassShared* Engine::sharedClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_sharedClass;
}

Class* Engine::moduleClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_moduleClass;
}

SynClasses* Engine::synclasses() const
{
   fassert( m_instance != 0 );
   return m_instance->m_synClasses;
}


Symbol* Engine::baseSymbol() const 
{
   return m_baseSymbol;
}


Symbol* Engine::getSymbol( const String& name, bool global )
{
   fassert( m_instance != 0 );
   return m_instance->m_symbols->get(name, global ? 1 : 0);
}

Symbol* Engine::getSymbol( const String& name, bool global, bool& isFirst )
{
   fassert( m_instance != 0 );
   return m_instance->m_symbols->get(name, global ? 1 : 0, isFirst);
}

void Engine::refSymbol( Symbol* sym )
{
   fassert( m_instance != 0 );
   m_instance->m_symbols->ref(sym, sym->isGlobal() ? 1 : 0);
}

void Engine::releaseSymbol( Symbol* sym )
{
   fassert( m_instance != 0 );
   m_instance->m_symbols->release(sym, sym->isGlobal() ? 1 : 0);
}

}

/* end of engine.cpp */

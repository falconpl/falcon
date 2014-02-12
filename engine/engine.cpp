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
#include <falcon/stdhandlers.h>
#include <falcon/error.h>
#include <falcon/classes/classerror.h>
#include <falcon/stderrors.h>

#include <falcon/classes/classnil.h>
#include <falcon/classes/classbool.h>
#include <falcon/classes/classint.h>
#include <falcon/classes/classnumeric.h>
#include <falcon/classes/classmethod.h>


//--- Virtual file systems ---
#include <falcon/vfs_file.h>

//--- standard transcoder headers ---

#include <falcon/tc/transcoderc.h>
#include <falcon/tc/transcoderf16.h>
#include <falcon/tc/transcoderf32.h>
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
#include <falcon/builtin/foreach.h>
#include <falcon/builtin/render.h>
#include <falcon/builtin/tostring.h>
#include <falcon/builtin/dynprop.h>

#include <falcon/bom.h>
#include <falcon/stdsteps.h>
#include <falcon/synfunc.h>
#include <falcon/gclock.h>

//--- object headers ---
#include <falcon/pseudofunc.h>
#include <falcon/collector.h>
#include <falcon/synclasses.h>

#include <falcon/psteps/exprinherit.h>
#include <falcon/psteps/exprcase.h>

#include <falcon/prototypeclass.h>

#include <falcon/modspace.h>
#include <falcon/symbol.h>

#include <falcon/item.h>         // for builtin
#include <falcon/pdata.h>

#include <falcon/stdmpxfactories.h>
#include <falcon/sys.h>

#include <falcon/config.h>

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

   inline Symbol* get(const String& name)
   {
      Symbol *s;
      m_mtx.lock();
      SymbolSet::iterator iter = m_symbols.find(&name);
      if( iter == m_symbols.end() ) {
         s = new Symbol( name );
         m_symbols[&s->name()] = s;
      }
      else {
         s = iter->second;
         s->m_counter++;
      }
      m_mtx.unlock();

      return s;
   }

   inline Symbol* getNoRef(const String& name)
   {
      Symbol *s = 0;
      m_mtx.lock();
      SymbolSet::iterator iter = m_symbols.find(&name);
      if( iter != m_symbols.end() ) {
         s = iter->second;
      }
      m_mtx.unlock();

      return s;
   }

   inline void release( Symbol* s ) {
     m_mtx.lock();
     if( --s->m_counter == 0 ) {
        m_symbols.erase(&s->name());
        m_mtx.unlock();

        delete s;
     }
     else {
        m_mtx.unlock();
     }

   }

   inline void ref( Symbol* s ) {
     m_mtx.lock();
     s->m_counter++;
     m_mtx.unlock();
   }

   ~ SymbolPool() {
      SymbolSet::iterator iter = m_symbols.begin();
      SymbolSet::iterator end = m_symbols.end();

      while( iter != end ) {
         Symbol *sym = iter->second;
         if( --sym->m_counter == 0 )
         {
            delete sym;
         }
         ++iter;
      }
   }

private:
   class StringPtrCheck {
   public:
      inline bool operator ()( const String *s1, const String *s2 ) const { return *s1 < *s2; }
   };

   typedef std::map<const String*, Symbol*, StringPtrCheck> SymbolSet;

   SymbolSet m_symbols;
   Mutex m_mtx;
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

   m_rand.seedWithPid();
   m_pdata = new PData;
   m_mtx = new Mutex;
   m_log = new Log;

   m_collector = new Collector;

   m_mantras = new MantraMap;
   m_stdHandlers = new StdHandlers();
   m_errHandlers = new MantraMap;

   //=====================================
   // Standard file systems.
   //
   m_vfs.addVFS("", new VFSFile );
   m_vfs.addVFS("file", new VFSFile );


   //=====================================
   // Initialization of standard deep types.
   //
   m_symbols = new SymbolPool;
   m_baseSymbol = m_symbols->get("$base");
   m_ruleBaseSymbol = m_symbols->get("$rulebase");

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
   m_pools = new PoolList;

   //=====================================
   // Adding standard transcoders.
   //

   m_tcoders = new TranscoderMap;
   addTranscoder( new TranscoderC );
   addTranscoder( new TranscoderF16 );
   addTranscoder( new TranscoderF32 );
   addTranscoder( new TranscoderUTF8 );

   //============================================
   // Creating predefined symbols
   //
   m_predefs = new PredefMap;
   m_stdSteps = new StdSteps;

   //=====================================
   // Adding standard pseudo functions.
   //
   m_stdHandlers->subscribe( this );
   // We can now update the item init
   Item::init(this);


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
   addMantra(new Ext::Render);
   addMantra(new Ext::Get);
   addMantra(new Ext::Set);
   addMantra(new Ext::Has);
   addMantra(new Ext::Properties);
   addMantra(new Ext::Approp);
   addMantra(new Ext::Foreach);

   //============================================
   // Creating singletons
   //

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

   //=====================================
   // Syntax Reflection
   //
   
   m_synClasses = new SynClasses(m_stdHandlers->syntreeClass(), m_stdHandlers->statementClass(), m_stdHandlers->expressionClass() );
   m_synClasses->subscribe( this );
   
   //=====================================
   // File/stream i/o
   //
   m_stdStreamTraits = new StdMpxFactories;

   //=====================================
   // The Core Module
   //
   m_core  = new CoreModule;

   //=====================================
   // The standard errors
   //

   // need the base class ...
   addMantra( registerError(new ClassError("Error", false) ) );

   // and all the others...
   #define FALCON_IMPLEMENT_ENGINE_ERRORS
   #include <falcon/stderrors.h>
      
   MESSAGE( "Engine creation complete" );
}


Engine::~Engine()
{
   MESSAGE( "Engine destruction started" );

   // remove all unneeded memory
   delete m_collector;

   /** Bye bye core... */
   m_core->decref();
   
   // ===============================
   // DO NOT Delete standard item classes -- they are mantras
   //

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
   delete m_errHandlers;
   
   // ===============================
   // delete builtin symbols
   //   
   delete m_predefs;   
   delete m_synClasses;
   delete m_stdHandlers;
   
   //============================================
   // Delete singletons
   //
   delete m_bom;
   delete m_stdSteps;
   delete m_pdata;

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


   // Delete symbols for last
   delete m_symbols;

   delete m_log;
   MESSAGE( "Engine destroyed" );
}

void Engine::init()
{
   MESSAGE( "Engine init()" );
   if( m_instance == 0 )
   {
      m_instance = new Engine;
      m_instance->collector()->start();
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

#if FALCON_TRACE_GC
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
#endif

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


Class* Engine::registerError( Class* errorClass )
{
   m_mtxEH.lock();
   (*m_errHandlers)[errorClass->name()] = errorClass;
   m_mtxEH.unlock();
   return errorClass;
}


void Engine::unregisterError( Class* errorClass )
{
   m_mtxEH.lock();
   m_errHandlers->erase(errorClass->name());
   m_mtxEH.unlock();
}


Class* Engine::getError( const String& name ) const
{
   Class* err = 0;
   m_mtxEH.lock();
   MantraMap::const_iterator pos = m_errHandlers->find(name);
   if( pos != m_errHandlers->end() )
   {
      err = static_cast<Class*>(pos->second);
   }
   m_mtxEH.unlock();

   if( err == 0 )
   {
      printf("YAY\n");
   }
   return err;
}

const String& Engine::version() const
{
   static String sv( String(FALCON_VERSION) + " " + FALCON_VERSION_SPEC + "(" + FALCON_VERSION_NAME +")" );
   return sv;
}

const String& Engine::fullVersion() const
{
   static String sv( String(FALCON_VERSION) + " " + String(FALCON_VERSION_SPEC) + "(" + FALCON_VERSION_NAME +") build " + String().N(FALCON_VERSION_BUILD_ID) );
   return sv;
}


int64 Engine::versionID() const
{
   return FALCON_VERSION_NUM;
}

//=====================================================
// Type handlers
//

Class* Engine::getTypeClass( int type )
{
   PARANOID("type out of range", (type < FLC_ITEM_COUNT) );
   return m_classes[type];
}


StdHandlers* Engine::handlers()
{
   fassert( m_instance != 0 );
   return m_instance->m_stdHandlers;
}

StdMpxFactories* Engine::mpxFactories()
{
   fassert( m_instance != 0 );
   return m_instance->m_stdStreamTraits;
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

Symbol* Engine::ruleBaseSymbol() const
{
   return m_ruleBaseSymbol;
}

Symbol* Engine::getSymbol( const String& name )
{
   fassert( m_instance != 0 );
   return m_instance->m_symbols->get(name);
}

Symbol* Engine::getSymbolNoRef( const String& name )
{
   fassert( m_instance != 0 );
   return m_instance->m_symbols->getNoRef(name);
}

void Engine::refSymbol( Symbol* sym )
{
   fassert( m_instance != 0 );
   m_instance->m_symbols->ref(sym );
}

void Engine::releaseSymbol( Symbol* sym )
{
   fassert( m_instance != 0 );
   m_instance->m_symbols->release(sym);
}




}

/* end of engine.cpp */

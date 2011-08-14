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

#include <map>
#include <stdio.h>

#include <falcon/trace.h>
#include <falcon/engine.h>

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/string.h>
#include <falcon/mt.h>


//--- Virtual file systems ---
#include <falcon/vfs_file.h>

//--- standard transcoder headers ---

#include <falcon/transcoderc.h>
#include <falcon/transcoderutf8.h>

//--- core function headers ---
#include <falcon/cm/coremodule.h>
#include <falcon/cm/compare.h>
#include <falcon/cm/len.h>
#include <falcon/cm/minmax.h>
#include <falcon/cm/typeid.h>
#include <falcon/cm/clone.h>

#include <falcon/bom.h>
#include <falcon/stdsteps.h>

//--- object headers ---
#include <falcon/pseudofunc.h>
#include <falcon/collector.h>

//--- type headers ---
#include <falcon/classfunction.h>
#include <falcon/classnil.h>
#include <falcon/classbool.h>
#include <falcon/classint.h>
#include <falcon/classnumeric.h>
#include <falcon/classstring.h>
#include <falcon/classrange.h>
#include <falcon/classarray.h>
#include <falcon/classdict.h>
#include <falcon/classgeneric.h>
#include <falcon/classnumeric.h>
#include <falcon/classmethod.h>
#include <falcon/prototypeclass.h>
#include <falcon/metaclass.h>
#include <falcon/classreference.h>

#include <falcon/stderrors.h>

#include <falcon/paranoid.h>
#include <map>


namespace Falcon
{


//=======================================================
// Private classes known by the engine -- Utils
//

class TranscoderMap: public std::map<String, Transcoder*>
{
};

class PseudoFunctionMap: public std::map<String, PseudoFunction*>
{
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

   m_mtx = new Mutex;
   m_collector = new Collector;

   //=====================================
   // Standard file systems.
   //
   m_vfs.addVFS("", new VFSFile );
   m_vfs.addVFS("file", new VFSFile );


   //=====================================
   // Initialization of standard deep types.
   //
   m_functionClass = new ClassFunction;
   m_stringClass = new ClassString;
   m_rangeClass = new ClassRange;
   m_arrayClass = new ClassArray;
   m_dictClass = new ClassDict;
   m_protoClass = new PrototypeClass;
   m_metaClass = new MetaClass;
   m_genericClass = new ClassGeneric;
   m_referenceClass = new ClassReference;

   // Initialization of the class vector.
   m_classes[FLC_ITEM_NIL] = new ClassNil;
   m_classes[FLC_ITEM_BOOL] = new ClassBool;
   m_classes[FLC_ITEM_INT] = new ClassInt;
   m_classes[FLC_ITEM_NUM] = new ClassNumeric;
   m_classes[FLC_ITEM_FUNC] = new ClassFunction;
   m_classes[FLC_ITEM_METHOD] = new ClassMethod;
   m_classes[FLC_ITEM_REF] = new ClassReference;   

   m_bom = new BOM;

   //=====================================
   // Adding standard transcoders.
   //

   m_tcoders = new TranscoderMap;
   addTranscoder( new TranscoderC );
   addTranscoder( new TranscoderUTF8 );

   //=====================================
   // Adding standard pseudo functions.
   //

   m_tpfuncs = new PseudoFunctionMap;
   addPseudoFunction(new Ext::Compare);
   addPseudoFunction(new Ext::Len);
   addPseudoFunction(new Ext::Max);
   addPseudoFunction(new Ext::Min);
   addPseudoFunction(new Ext::TypeId);
   addPseudoFunction(new Ext::Clone);

   //============================================
   // Creating singletons
   //
   
   m_stdSteps = new StdSteps;
   m_stdErrors = new StdErrors;
   
   //=====================================
   // The Core Module
   //
   m_instance = this; // modules need the engine.
   m_core  = new CoreModule;

   MESSAGE( "Engine creation complete" );
}


Engine::~Engine()
{
   MESSAGE( "Engine destruction started" );

   m_collector->stop();

   delete m_stringClass;
   delete m_rangeClass;
   delete m_arrayClass;
   delete m_dictClass;
   delete m_protoClass;
   delete m_metaClass;
   delete m_functionClass;
   delete m_genericClass;
   delete m_referenceClass;

   // ===============================
   // Delete standard item classes
   //
   for ( int count = 0; count < FLC_ITEM_COUNT; ++count )
   {
      delete m_classes[count];
   }

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

   //============================================
   // Delete singletons
   //
   delete m_core;
   delete m_bom;
   delete m_stdSteps;
   delete m_stdErrors;

   delete m_collector;
   delete m_mtx;

   MESSAGE( "Engine destroyed" );
}

void Engine::init()
{
   MESSAGE( "Engine init()" );
   fassert( m_instance == 0 );
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

bool Engine::addPseudoFunction( PseudoFunction* pf )
{
   m_mtx->lock();
   PseudoFunctionMap::iterator iter = m_tpfuncs->find(pf->name());
   if ( iter != m_tpfuncs->end() )
   {
      m_mtx->unlock();
      return false;
   }

   (*m_tpfuncs)[pf->name()] = pf;
   m_mtx->unlock();
   return true;
}

PseudoFunction* Engine::getPseudoFunction( const String& name ) const
{
   m_mtx->lock();
   PseudoFunctionMap::iterator iter = m_tpfuncs->find(name);
   if ( iter == m_tpfuncs->end() )
   {
      m_mtx->unlock();
      return 0;
   }

   PseudoFunction* ret = iter->second;
   m_mtx->unlock();
   return ret;
}

//=====================================================
// Global objects
//

Engine* Engine::instance()
{
   fassert( m_instance != 0 );
   return m_instance;
}

 
Collector* Engine::collector() const
{
   fassert( m_instance != 0 );
   return m_instance->m_collector;
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

Class* Engine::genericClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_genericClass;
}

ClassReference* Engine::referenceClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_referenceClass;
}

}

/* end of engine.cpp */

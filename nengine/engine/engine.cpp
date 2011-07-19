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

#include <falcon/bom.h>

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
#include <falcon/classarray.h>
#include <falcon/classdict.h>
#include <falcon/classnumeric.h>
#include <falcon/prototypeclass.h>
#include <falcon/metaclass.h>

//--- error headers ---
#include <falcon/accesserror.h>
#include <falcon/accesstypeerror.h>
#include <falcon/errorclass.h>
#include <falcon/codeerror.h>
#include <falcon/genericerror.h>
#include <falcon/interruptederror.h>
#include <falcon/ioerror.h>
#include <falcon/operanderror.h>
#include <falcon/unsupportederror.h>
#include <falcon/syntaxerror.h>
#include <falcon/encodingerror.h>

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
// Private classes known by the engine
//
class AccessErrorClass: public ErrorClass
{
public:
   AccessErrorClass():
      ErrorClass("AccessError")
   {}

   virtual void* create(void* creationParams ) const
   {
      return new AccessError( *static_cast<ErrorParam*>(creationParams) );
   }
};

class AccessTypeErrorClass: public ErrorClass
{
public:
   AccessTypeErrorClass():
      ErrorClass("AccessTypeError")
   {}

   virtual void* create(void* creationParams ) const
   {
      return new AccessTypeError( *static_cast<ErrorParam*>(creationParams) );
   }
};


class CodeErrorClass: public ErrorClass
{
public:
   CodeErrorClass():
      ErrorClass("CodeError")
   {}

   virtual void* create(void* creationParams ) const
   {
      return new CodeError( *static_cast<ErrorParam*>(creationParams) );
   }
};

class GenericErrorClass: public ErrorClass
{
public:
   GenericErrorClass():
      ErrorClass( "GenericError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new GenericError( *static_cast<ErrorParam*>(creationParams) );
   }
};

class InterruptedErrorClass: public ErrorClass
{
public:
   InterruptedErrorClass():
      ErrorClass( "InterruptedError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new InterruptedError( *static_cast<ErrorParam*>(creationParams) );
   }
};


class IOErrorClass: public ErrorClass
{
public:
   IOErrorClass():
      ErrorClass( "IOError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new IOError( *static_cast<ErrorParam*>(creationParams) );
   }
};


class OperandErrorClass: public ErrorClass
{
public:
   OperandErrorClass():
      ErrorClass( "OperandError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new OperandError( *static_cast<ErrorParam*>(creationParams) );
   }
};


class UnsupportedErrorClass: public ErrorClass
{
public:
   UnsupportedErrorClass():
      ErrorClass( "UnsupportedError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new UnsupportedError( *static_cast<ErrorParam*>(creationParams) );
   }
};


class EncodingErrorClass: public ErrorClass
{
public:
   EncodingErrorClass():
      ErrorClass( "EncodingError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new EncodingError( *static_cast<ErrorParam*>(creationParams) );
   }
};

class SyntaxErrorClass: public ErrorClass
{
public:
   SyntaxErrorClass():
      ErrorClass( "SyntaxError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new SyntaxError( *static_cast<ErrorParam*>(creationParams) );
   }
};

class ParamErrorClass: public ErrorClass
{
public:
   ParamErrorClass():
      ErrorClass( "ParamError" )
      {}

   virtual void* create(void* creationParams ) const
   {
      return new SyntaxError( *static_cast<ErrorParam*>(creationParams) );
   }
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
   m_arrayClass = new ClassArray;
   m_dictClass = new ClassDict;
   m_protoClass = new PrototypeClass;
   m_classClass = new MetaClass;

   // Initialization of the class vector.
   m_classes[FLC_ITEM_NIL] = new ClassNil;
   m_classes[FLC_ITEM_BOOL] = new ClassBool;
   m_classes[FLC_ITEM_INT] = new ClassInt;
   m_classes[FLC_ITEM_NUM] = new ClassNumeric;
   m_classes[FLC_ITEM_FUNC] = new ClassFunction;
   m_classes[FLC_ITEM_METHOD] = new ClassNil;
   m_classes[FLC_ITEM_BASEMETHOD] = new ClassNil;

   //=====================================
   // Initialization of standard errors.
   //
   m_accessErrorClass = new AccessErrorClass;
   m_accessTypeErrorClass = new AccessTypeErrorClass;
   m_codeErrorClass = new CodeErrorClass;
   m_genericErrorClass = new GenericErrorClass;
   m_interruptedErrorClass = new InterruptedErrorClass;
   m_ioErrorClass = new IOErrorClass;
   m_operandErrorClass = new OperandErrorClass;
   m_unsupportedErrorClass = new UnsupportedErrorClass;
   m_encodingErrorClass = new EncodingErrorClass;
   m_syntaxErrorClass = new SyntaxErrorClass;
   m_paramErrorClass = new ParamErrorClass;

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
   delete m_arrayClass;
   delete m_dictClass;
   delete m_protoClass;
   delete m_classClass;
   delete m_functionClass;

   // ===============================
   // Delete standard error classes
   //
   delete m_accessErrorClass;
   delete m_accessTypeErrorClass;
   delete m_codeErrorClass;
   delete m_genericErrorClass;
   delete m_ioErrorClass;
   delete m_interruptedErrorClass;
   delete m_operandErrorClass;
   delete m_unsupportedErrorClass;
   delete m_encodingErrorClass;
   delete m_paramErrorClass;

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

Class* Engine::classClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_classClass;
}

//=====================================================
// Error handlers
//

Class* Engine::codeErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_codeErrorClass;
}

Class* Engine::genericErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_genericErrorClass;
}

Class* Engine::ioErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_ioErrorClass;
}

Class* Engine::interruptedErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_interruptedErrorClass;
}

Class* Engine::encodingErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_encodingErrorClass;
}

Class* Engine::accessErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_accessErrorClass;
}

Class* Engine::accessTypeErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_accessTypeErrorClass;
}

Class* Engine::syntaxErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_syntaxErrorClass;
}

Class* Engine::paramErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_paramErrorClass;
}


Class* Engine::operandErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_operandErrorClass;
}

Class* Engine::unsupportedErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_unsupportedErrorClass;
}

}

/* end of engine.cpp */

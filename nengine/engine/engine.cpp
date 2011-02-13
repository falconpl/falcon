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

#include <falcon/engine.h>

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/string.h>
#include <falcon/mt.h>

#include <falcon/trace.h>
//--- object headers ---

#include <falcon/collector.h>

//--- utility headers ---
#include <falcon/errorclass.h>
#include <falcon/codeerror.h>
#include <falcon/genericerror.h>

#include <falcon/corefunction.h>

namespace Falcon
{
//=======================================================
// Private classes known by the engine
//

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

//=======================================================
// Engine static declarations
//

Engine* Engine::m_instance = 0;

//=======================================================
// Engine implementation
//

Engine::Engine()
{
   TRACE("Engine creation started", 0 )
   #ifdef FALCON_SYSTEM_WIN
   m_bWindowsNamesConversion = true;
   #else
   m_bWindowsNamesConversion = false;
   #endif

   m_mtx = new Mutex;
   m_collector = new Collector;


   //=====================================
   // Initialization of standard deep types.
   //
   m_functionClass = new CoreFunction;

   //=====================================
   // Initialization of standard errors.
   //
   m_codeErrorClass = new CodeErrorClass;
   m_genericErrorClass = new GenericErrorClass;

   TRACE("Engine creation complete", 0 )
}

Engine::~Engine()
{
   TRACE("Engine destruction started", 0 )
   delete m_mtx;
   delete m_collector;
   delete m_codeErrorClass;
   TRACE("Engine destroyed", 0 )
}

void Engine::init()
{
   TRACE("Engine init()", 0 )
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
   TRACE("Engine shutdown started", 0 )
   fassert( m_instance != 0 );
   if( m_instance != 0 )
   {
      // TODO
      // m_instance->collector()->start();

      delete m_instance;
      m_instance = 0;
      TRACE("Engine shutdown complete", 0 )
   }
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

Class* Engine::functionClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_functionClass;
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

}

/* end of engine.cpp */

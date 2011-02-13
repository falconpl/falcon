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

//--- object headers ---

#include <falcon/collector.h>

//--- utility headers ---
#include <falcon/errorclass.h>
#include <falcon/codeerror.h>

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


//=======================================================
// Engine static declarations
//

Engine* Engine::m_instance = 0;

//=======================================================
// Engine implementation
//

Engine::Engine()
{
   #ifdef FALCON_SYSTEM_WIN
   m_bWindowsNamesConversion = true;
   #else
   m_bWindowsNamesConversion = false;
   #endif

   m_mtx = new Mutex;
   m_collector = new Collector;

   m_codeErrorClass = new CodeErrorClass;
}

Engine::~Engine()
{
   delete m_mtx;
   delete m_collector;
   delete m_codeErrorClass;
}

void Engine::init()
{
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
   fassert( m_instance != 0 );
   if( m_instance != 0 )
   {
      // TODO
      // m_instance->collector()->start();

      delete m_instance;
      m_instance = 0;
   }
}

Engine* Engine::instance()
{
   fassert( m_instance != 0 );
   return m_instance;
}

bool Engine::isWindows() const
{
   fassert( m_instance != 0 );
   m_instance->m_bWindowsNamesConversion;
}

 
Collector* Engine::collector() const
{
   fassert( m_instance != 0 );
   return m_instance->m_collector;
}

Class* Engine::codeErrorClass() const
{
   fassert( m_instance != 0 );
   return m_instance->m_codeErrorClass;
}


}

/* end of engine.cpp */

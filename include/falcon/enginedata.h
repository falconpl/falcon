/*
   FALCON - The Falcon Programming Language.
   FILE: enginedata.h

   Definition of engine static data used by falcon engine.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun dic 4 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Definition of static data used by falcon engine.
   This data must be passed to modules so that they can configure
   global variables that are used by Falcon engine functions.
*/

#ifndef flc_enginedata_H
#define flc_enginedata_H

#include <falcon/setup.h>
#include <falcon/memory.h>
#include <falcon/engstrings.h>
#include <falcon/strtable.h>
#include <stdlib.h>

namespace Falcon {

namespace Engine {
/** Engine data that is available for modules to be changed.

   This class holds atomic and mutex-lock pointer functions, as well as the
   string declaring the standard engine language (and in future, the engine
   locale).

   Each module holds a pointer to a singletone instance of this structure, which
   is created at engine initialization. Is then possible for
   each module, even if loaded after the initialization of the engine,
   to change the fields in this structure, and have this reflected into all
*/
class RWData
{
public:
   volatile long (*atomicInc)( volatile long &data );
   volatile long (*atomicDec)( volatile long &data );
   void (*lockEngine)();
   void (*unlockEngine)();
   String language;
};

extern FALCON_DYN_SYM RWData *rwdata;

/** Shortcut function */
inline volatile long atomicInc( volatile long &data )
{
   return rwdata->atomicInc( data );   
}

inline volatile long atomicDec( volatile long &data )
{
   return rwdata->atomicDec( data );   
}

inline void lockEngine()
{
   rwdata->lockEngine();   
}

inline void unlockEngine()
{
   rwdata->unlockEngine();   
}

inline const String &language()
{
   return rwdata->language;
}

inline void language( const String &lang )
{
   rwdata->language = lang;
}

}


/** Class used to transport automatically engine data into modules.

   An instance of this class must be created in the main program using the Falcon Engine
   after configuring the Falcon engine global variables:

   - memAlloc: pointer to the function allocating memory
   - memFree: pointer to the function deallocating memory
   - memRealloc: pointer to the function changing memory allocation
   - engineStrings: pointer to the global engine string table
   - rwdata: Pointer to the RW data (actually, created in the engine)

   The main program should set this variables as it prefers; then the Falcon::Init() function
   should be called to ensure that the DLL holding the engine is updated. As an instance
   of EngineData created by the engine will be passed to the modules loaded by the loaders,
   this should be done before any module is loaded.
*/
class FALCON_DYN_CLASS EngineData
{
   void * (*m_memAlloc) ( size_t );
   void (*m_memFree) ( void * );
   void * (*m_memRealloc) ( void *,  size_t );
   StringTable *m_engineStrings;
   Engine::RWData *m_rwData;

public:

   /** Engine data constructor.
      As the constructor is granted to be inline, the data from Falcon Engine is
      taken in the code of the caller; in example, it records the values that the
      main program has set for engine static variables. Then, this instance can
      be passed around to configure modules and Falcon engine dll as the main
      program wishes.
   */
   EngineData()
   {
      initRWData();
      get();
   }

   /** Loads engine variables in this instance.
      This method loads the Falcon engine static variables so that they
      can be sent around. The main program should set global engine
      variables according to its own needs and then pass around an
      instance of the EngineData() class so that modules and
      engines can configure themselves.

      The method is inline, so it is granted to load the variables
      from the static context of the caller.
   */
   inline void get()
   {
	   m_memAlloc = ::Falcon::memAlloc;
      m_memFree = ::Falcon::memFree;
      m_memRealloc = ::Falcon::memRealloc;
      m_engineStrings = ::Falcon::engineStrings;
      m_rwData = ::Falcon::Engine::rwdata;
   }

   /** Sets global engine variables.
      Call this method inside modules that receive an instance of this
      class to setup global engine varialbes according to the ones
      in the main program.
   */
   void set() const;

   /** Create the initial. */
   void initRWData();
};


/** Initialize the main engine DLL using global data prepared by the main program. 
 
*/
extern "C" FALCON_DYN_SYM void Init( const EngineData &data );

}

#endif

/* end of enginedata.h */

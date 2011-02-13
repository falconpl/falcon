/*
   FALCON - The Falcon Programming Language.
   FILE: engine.h

   Global variables known by the falcon System.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 12:25:12 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_ENGINE_H_
#define	_FALCON_ENGINE_H_

#include <falcon/setup.h>

namespace Falcon
{
class Class;
class Collector;
class Mutex;

/** Falcon application global data.

 This class stores the gloal items that must be known by the falcon engine
 library, and starts the subsystems needed by Falcon to handle application-wide
 objects.

 An application is required to call Engine::init() method when the falcon engine
 is first needed, and to call Engine::shutdown() before exit.

 Various assert points are available in debug code to check for correct
 initialization sequence, but they will be removed in release code, so if the
 engine is not properly initialized you may expect random crashes in release
 (right near the first time you use Falcon stuff).

 @note init() and shutdown() code are not thread-safe. Be sure to invoke them
 in a single-thread context.
 
 */
class FALCON_DYN_SYM Engine
{
public:

   /** Initializes the Falcon subsystem. */
   static void init();

   /** Terminates the Falcon subsystem. */
   static void shutdown();

   /** Returns the current engine instance.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   static Engine* instance();

   //==========================================================================
   // Global Settings
   //

   /** True when running on windows system.
    
    File naming convention and other details are different on windows systems.
    */
   bool isWindows() const;

   //==========================================================================
   // Global Objects
   //

   /** The global collector.
    */
   Collector* collector() const;

   /** Returns the global instance of the CodeError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* codeErrorClass() const;

    /** Returns the global instance of the Function class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* functionClass() const;


protected:
   Engine();
   ~Engine();

   static Engine* m_instance;
   Mutex* m_mtx;
   Collector* m_collector;

   //===============================================
   // Global settings
   //
   bool m_bWindowsNamesConversion;

   //===============================================
   // Global object handlers
   //
   Class* m_codeErrorClass;
   Class* m_functionClass;
};

}

#endif	/* _FALCON_ENGINE_H_ */

/* end of engine.h */

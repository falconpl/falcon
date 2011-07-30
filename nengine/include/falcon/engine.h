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
#define _FALCON_ENGINE_H_

#include <falcon/setup.h>
#include <falcon/itemid.h>
#include <falcon/string.h>
#include <falcon/vfsiface.h>

#include <falcon/vfsprovider.h>

namespace Falcon
{
class Class;
class Collector;
class Mutex;
class Transcoder;
class TranscoderMap;
class PseudoFunction;
class PseudoFunctionMap;
class Module;
class BOM;
class ClassReference;
class StdSteps;

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

   /** Terminates the program NOW with an error message.
    \param msg The message that will be displayed on termination.
    */
   static void die( const String& msg );

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

   /** Return the class handing the base type reflected by this item type ID.
    \param type The type of the items for which the handler class needs to be known.
    \return The handler class, or 0 if the type ID is not defined or doesn't have an handler class.

    This method returns the class that, if provided in a User or Deep item as
    handler class, would cause the item to be treated correctly.
    */
   Class* getTypeClass( int type );

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

   //==========================================================================
   // Type handlers
   //

   /** Returns the global instance of the Function class.
   \return the Engine instance of the Function Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* functionClass() const;

   /** Returns the global instance of the String class.
   \return the Engine instance of the String Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* stringClass() const;

   /** Returns the global instance of the Array class.
   \return the Engine instance of the Array Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* arrayClass() const;

   /** Returns the global instance of the Dictionary class.
   \return the Engine instance of the Dictionary Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* dictClass() const;

   /** Returns the global instance of the Prototype class.
   \return the Engine instance of the Prototype Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* protoClass() const;

   /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaClass() const;

   /** Returns the global instance of the ClassReference class.
   \return the Engine instance of the ClassReference handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassReference* referenceClass() const;
   
   //==========================================================================
   // Error handlers
   //

   /** Returns the global instance of the CodeError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* codeErrorClass() const;

   /** Returns the global instance of the GenericError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* genericErrorClass() const;

   /** Returns the global instance of the OperandError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* operandErrorClass() const;

   /** Returns the global instance of the UnsupportedError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* unsupportedErrorClass() const;

   /** Returns the global instance of the IOError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* ioErrorClass() const;

   /** Returns the global instance of the InterruptedError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* interruptedErrorClass() const;

   /** Returns the global instance of the EncodingError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* encodingErrorClass() const;

   /** Returns the global instance of the AccessError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* accessErrorClass() const;

   /** Returns the global instance of the AccessTypeError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* accessTypeErrorClass() const;

   /** Returns the global instance of the SyntaxError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* syntaxErrorClass() const;

   /** Returns the global instance of the ParamError class.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* paramErrorClass() const;

   /** Adds a transcoder to the engine.
    \param A new transcoder to be registered in the engine.
    \return true if the transcoder can be added, false if it was already registered.

    The owenrship of the transcoder is passed to the engine. The transcoder will
    be destroyed at engine destruction.
   */

   bool addTranscoder( Transcoder* enc );

   /** Gets a transcoder.
    \param name The ANSI encoding name of the encoding served by the desired transcoder.
    \return A valid transcoder pointer or 0 if the encoding is unknown.
    */
   Transcoder* getTranscoder( const String& name );

   /* Get a transcoder that will serve the current system text encoding.
    \param bDefault if true, a "C" default transcoder will be returned incase the
           system encoding cannot be found
    \return A valid transcoder pointer or 0 if the encoding is unknown.

    TODO
    */
   //Transcoder* getSystemTranscoder( bool bDefault = false );

   VFSIface& vsf() { return m_vfs; }
   const VFSIface& vsf() const { return m_vfs; }

   /** Adds a pseudofunction to the engine.
    \param pf The Pseudo function to be added.
    \return False if another pseudofunction with the same name was already added.

    Pseudo-functions act as built-in with respect to the Falcon source compiler;
    if found in a call-expression they are substituted with a virtual machine
    operation on the spot. If accessed in any other way, they behave as
    normal functions.

    Notice that pseudofunction appare as global symbols in the global context;
    however, it is possible to create namespaced pseudofunctions setting a
    name prefix in their name.

    \note the ownership of the pseudofunction is passed to the engine.
    */
   bool addPseudoFunction( PseudoFunction* pf );

   /** Returns a previously added pseudo function.
    \param name The name of the pseudofunction to be searched.
    \return The pseudofunction coresponding with that name, or 0 if not found.
    
    \see addPseudoFunction
    */
   PseudoFunction* getPseudoFunction( const String& name ) const;

   /** Returns an instance of the core module that is used as read-only.
    \return A singleton instance of the core module.
    
    This instance is never linked in any VM and is used as a resource for
    undisposeable function (as i.e. the BOM methods).
    */
   Module* getCore() const { return m_core; }

   /** Returns the Basic Object Model method collection.
    \return A singleton instance of the core BOM handler collection.
   */
   BOM* getBom() const { return m_bom; }

   /** Archive of standard steps. */
   StdSteps* stdSteps() const { return m_stdSteps; }
   
protected:
   Engine();
   ~Engine();

   static Engine* m_instance;
   Mutex* m_mtx;
   Collector* m_collector;
   Class* m_classes[FLC_ITEM_COUNT];

   VFSIface m_vfs;
   //===============================================
   // Global settings
   //
   bool m_bWindowsNamesConversion;

   //===============================================
   // Global type handlers
   //
   Class* m_functionClass;
   Class* m_stringClass;
   Class* m_arrayClass;
   Class* m_dictClass;
   Class* m_protoClass;
   Class* m_metaClass;
   ClassReference* m_referenceClass;
   
   //===============================================
   // Standard error handlers
   //
   Class* m_accessErrorClass;
   Class* m_accessTypeErrorClass;
   Class* m_codeErrorClass;
   Class* m_genericErrorClass;
   Class* m_operandErrorClass;
   Class* m_unsupportedErrorClass;
   Class* m_ioErrorClass;
   Class* m_interruptedErrorClass;
   Class* m_encodingErrorClass;
   Class* m_syntaxErrorClass;
   Class* m_paramErrorClass;
   

   //===============================================
   // Transcoders
   //
   TranscoderMap* m_tcoders;

   //===============================================
   // The core module.
   //
   Module* m_core;
   BOM* m_bom;

   PseudoFunctionMap* m_tpfuncs;
   StdSteps* m_stdSteps;
};

}

#endif	/* _FALCON_ENGINE_H_ */

/* end of engine.h */

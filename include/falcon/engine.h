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
#include <falcon/mantra.h>

namespace Falcon
{
class Class;
class Collector;
class Mutex;
class Transcoder;
class TranscoderMap;
class PoolList;
class Pool;
class ClassReference;
class ClassShared;
class ClassModule;
class GCLock;

class PredefMap;
class MantraMap;

class Module;
class BOM;
class ClassRawMem;
class StdSteps;
class StdErrors;
class ModSpace;
class Item;
class Symbol;

class SynClasses;
class VMContext;

class Log;

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
class FALCON_DYN_CLASS Engine
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
   static Collector* collector();

   static GCToken* GC_store( const Class* cls, void* data );
   static GCToken* GC_H_store( const Class* cls, void* data, const String& src, int line );
   static GCLock* GC_storeLocked( const Class* cls, void* data );
   static GCLock* GC_H_storeLocked( const Class* cls, void* data, const String& src, int line );
   static GCLock* GC_lock( const Item& item );
   static void GC_unlock( GCLock* lock );

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
   
   /** Returns the global instance of the String class.
   \return the Engine instance of the String Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* rangeClass() const;

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
   
    /** Returns the global instance of the MetaFalconClass class.
   \return the Engine instance of the MetaFalconClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaFalconClass() const;
    
   /** Returns the global instance of the MetaHyperClass class.
   \return the Engine instance of the MetaHyperClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaHyperClass() const;
   
   /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* mantraClass() const;
   
      /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* synFuncClass() const;
   
   /** Returns the global instance of the GenericClass class.
   \return the Engine instance of the GenericClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* genericClass() const;
   
   /** Returns the global instance of the ClassTreeStep class.
   \return the Engine instance of the ClassTreeStep handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* treeStepClass() const;
   
   /** Returns the global instance of the statement class.
   \return the Engine instance of the ClassStatement handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* statementClass() const;
   
   /** Returns the global instance of the ClassExpression class.
   \return the Engine instance of the ClassExpression handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* expressionClass() const;
   
   /** Returns the global instance of the ClassSynTree class.
   \return the Engine instance of the ClassSynTree handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* syntreeClass() const;
   
   /** Returns the global instance of the ClassSymbol class.
   \return the Engine instance of the ClassSymbol handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* symbolClass() const;

   /** Returns the global instance of the ClassStorer class.
   \return the Engine instance of the ClassStorer handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* storerClass() const;

   /** Returns the global instance of the ClassRestorer class.
   \return the Engine instance of the ClassRestorer handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* restorerClass() const;

   /** Returns the global instance of the ClassStream class.
   \return the Engine instance of the ClassStream handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* streamClass() const;

   /** Returns the global instance of the ClassClosure class.
   \return the Engine instance of the ClassClosure handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* closureClass() const;
   
   /** Returns the global instance of the ClassRawMem class.
   \return the Engine instance of the ClassRawMem handler.

    Method init() must have been called before.

    Notice that the ClassRawMem is not reflected in the language:
    is just used internally by the engine as it at disposal of embedders
    and third party module writers to write extensions that require to pass
    some raw data to the Falcon garbage collector.
    
    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */   
   ClassRawMem* rawMemClass() const;

   /** Returns the global instance of the ClassReference class.
   \return the Engine instance of the ClassReference handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassReference* referenceClass() const;

   /** Returns the global instance of the SharedClass class.
   \return the Engine instance of the SharedClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassShared* sharedClass() const;

   /** Returns the global instance of the ModuleClass class.
   \return the Engine instance of the ModuleClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* moduleClass() const;

   /** Returns the collection of standard syntactic tree classes.
   \return the Engine instance of the SynClasses class collection.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   SynClasses* synclasses() const;
   
   /** Returns the standard collection of error handlers.
    */
   StdErrors* stdErrors() const { return m_stdErrors; }

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

   VFSIface& vfs() { return m_vfs; }
   const VFSIface& vfs() const { return m_vfs; }

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
   
   
   /** Adds a builtin item.
    @param name The name under which the builtin is known.
    @param value The value that is to be published to the scripts.
    
    */
   bool addBuiltin( const String& name, const Item& value );
   
   /** Gets a pre-defined built-in value. 
    */
   const Item* getBuiltin( const String& name ) const;
   
   /** Centralized repository of publically available classes and functions.
    @param reg The mantra to be stored.
    \return True if the mantra name wasn't already registered, false otherwise.
    
    A Mantra is a invocation for the engine that is known by name. Usually
    it's a Class or a Function.
    
    This method controls a central repository for Mantras that are not provided
    by modules (i.e. built in). are not created by modules. It's the case of classes
    and functions provided by the engine or by embedding applications.
    
    Mantras stored in the engine class register are known only by name, and they
    must have a well-known system-wide uinque name. For instance, "Integer",
    "String" and so on. Class having a module might be added to the register;
    in that case, they will be known by their full ID (module logical name
    preceding their name, separated by a dot). 
    
    However, there isn't any way to unregister a class, so, when registering
    a class stored in a module, the caller must be sure that the module is static
    and will not be destroyed until the end of the process.
    
    The class register is not owning the stored classes; this is just a
    dictionary of well known classes that are not directly accessible in modules
    that a VM may be provided with. 
    
    \note This method automatically calls addBuiltin to create a item visible
    by the scripts. In other words, registered mantras become script-visible.
    
    \note Mantras added this way are owned by the engine; the engine destroys
    them at destruction.
    */
   bool addMantra( Mantra* reg );
   
   /** Gets a previously registered mantra by name.
    @param name The name or full class ID of the desired class.
    @param cat A category to filter out undesired mantras.
    @return The mantra if it had been registered, 0 if the name is not found.
    
    @see addMantra
    */
   Mantra* getMantra( const String& name, Mantra::t_category cat = Mantra::e_c_none ) const;

   /** Set the context run by this thread.
    \param ctx The context being run.
    Each time the VM changes the running context in the current VM execution
    thread, this method is called to update the global visibility of the context.
    */
   void setCurrentContext( VMContext* ctx );
   
   /** Adds an object-specific memory pool.
    
    The pool will be destroyed at engine exit.
    */
   void addPool( Pool* p );

   /** Returns a pointer to the base dynsymbol.
    
    The base dynsymbol is a symbol that is inserted
    at the head of each evaluation frame to save the corresponding
    data stack frame.
    */
   Symbol* baseSymbol() const;
   
   Log* log() const;

protected:
   Engine();
   ~Engine();

   static Engine* m_instance;
   Mutex* m_mtx;
   Collector* m_collector;
   Log* m_log;
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
   Class* m_rangeClass;
   Class* m_arrayClass;
   Class* m_dictClass;
   Class* m_protoClass;
   Class* m_metaClass;
   Class* m_metaFalconClass;
   Class* m_metaHyperClass;
   Class* m_mantraClass;
   Class* m_synFuncClass;
   Class* m_genericClass;

   ClassRawMem* m_rawMemClass;
   ClassReference* m_referenceClass;
   ClassShared* m_sharedClass;
   
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
   // Basic code reflection entities
   //
   Class* m_treeStepClass;
   Class* m_statementClass;
   Class* m_exprClass;
   Class* m_syntreeClass;
   Class* m_symbolClass;
   Class* m_closureClass;
   Class* m_storerClass;
   Class* m_restorerClass;
   Class* m_streamClass;
   ClassModule* m_moduleClass;
   
   SynClasses* m_synClasses;
   
   //===============================================
   // Transcoders
   //
   TranscoderMap* m_tcoders;
   
   //===============================================
   // Pools
   //
   PoolList* m_pools;

   //===============================================
   // The core module.
   //
   Module* m_core;
   BOM* m_bom;

   MantraMap* m_mantras;
   PredefMap* m_predefs;
   
   StdSteps* m_stdSteps;
   StdErrors* m_stdErrors;
   Symbol* m_baseSymbol;

};

}

#endif	/* _FALCON_ENGINE_H_ */

/* end of engine.h */

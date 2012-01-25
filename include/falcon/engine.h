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
class PredefMap;
class RegisteredClassesMap;
class MetaStorerMap;

class Module;
class BOM;
class ClassReference;
class StdSteps;
class StdErrors;
class ModSpace;
class Item;

class SynClasses;
class MetaStorer;
class VMContext;

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

   /** Returns a registered meta-storer. 
    \param name The required MetaStorer name.
    \return A pointer to a metastorer or 0 if not found. 
    \see MetaStorer
    */
   
   MetaStorer* getMetaStorer( const String& name );
   
   /** Records a storer.
    \param ms The storer.
    \return True if the storer was recorded, false if the storer name is already
    assigned.
    
    \see MetaStorer
    
    \note The engine becomes owner of the storer.
    */
   bool registerMetaStorer( MetaStorer* ms );
   
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
   
   
   /** Returns the global instance of the GenericClass class.
   \return the Engine instance of the GenericClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* genericClass() const;

   /** Returns the global instance of the ClassReference class.
   \return the Engine instance of the ClassReference handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassReference* referenceClass() const;
   
   
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
   
   /** Returns the global instance of the ClassDynSymbol class.
   \return the Engine instance of the ClassDynSymbol handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* dynSymbolClass() const;
   
   /** Returns the global instance of the ClassClosure class.
   \return the Engine instance of the ClassClosure handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* closureClass() const;
   
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
   
   /** Adds a builtin instance of this class.
    @param src The class of this item.
    
    This method adds a "class" item (Item with
    MetaClass as class, and this Class as value) to the set of common
    builtin items available to any falcon program.
    
    If the class has not a module, it is also automatically inserted
    into the class register via registerClass().    
    */
   bool addBuiltin( Class* src );
   
   /** Adds a builtin item.
    @param name The name under which the builtin is known.
    @param value The value that is to be published to the scripts.
    
    */
   bool addBuiltin( const String& name, const Item& value );
   
   /** Gets a pre-defined built-in value. 
    */
   const Item* getBuiltin( const String& name ) const;
   
   /** Centralized repository of publically available classes.
    @param cls The class to be stored.
    
    Classes are usually stored in the modules that create them; however, some 
    classes are not created by modules. It's the case of classe provided by
    the engine or by embedding applications.
    
    Usually, the scripts don't need those classes, but some processes may
    need to access this classes by name. For instance, serialization, or
    creation of standardized objects, may need to access a class knowing
    its name.
    
    Class stored in the engine class register are known only by name, and they
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
    
    */
   void registerClass( Class* reg );
   
   /** Gets a previously registered class by name.
    @param name The name or full class ID of the desired class.
    @return The class if it had been registered, 0 if the name is not found.
    
    @see registerClass
    */
   Class* getRegisteredClass( const String& name ) const;
   
   /** Returns the context currently active in the current tread.
    \return the context active in the current thread or 0 if the VM is not
    running any context.
    
    The context will be returned also if the VM active in the current thread
    is paused.
    */
   VMContext* currentContext() const;

   /** Set the context run by this thread.
    \param ctx The context being run.
    Each time the VM changes the running context in the current VM execution
    thread, this method is called to update the global visibility of the context.
    */
   void setCurrentContext( VMContext* ctx );
   
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
   Class* m_rangeClass;
   Class* m_arrayClass;
   Class* m_dictClass;
   Class* m_protoClass;
   Class* m_metaClass;
   Class* m_genericClass;
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
   // Basic code reflection entities
   //
   Class* m_treeStepClass;
   Class* m_statementClass;
   Class* m_exprClass;
   Class* m_syntreeClass;
   Class* m_symbolClass;
   Class* m_closureClass;
   
   SynClasses* m_synClasses;

   RegisteredClassesMap* m_regClasses;
   MetaStorerMap *m_metaStorers;
   
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
   PredefMap* m_predefs;
   
   StdSteps* m_stdSteps;
   StdErrors* m_stdErrors;
   
   // TODO: In MT, set this as TLS data.
   VMContext* m_currentContext;
};

}

#endif	/* _FALCON_ENGINE_H_ */

/* end of engine.h */

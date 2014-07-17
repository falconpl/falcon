/*
   FALCON - The Falcon Programming Language.
   FILE: stdhandlers.h

   Repository for standard handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 01 Mar 2013 19:41:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_STDHANDLERS_H_
#define _FALCON_STDHANDLERS_H_

#include <falcon/setup.h>
// who uses the handlers must use the engine
#include <falcon/engine.h>

namespace Falcon
{

class Class;
class ClassRawMem;
class ClassShared;
class ClassModule;
class Engine;

/**
 Class handling an array as an item in a falcon script.
 */

class FALCON_DYN_CLASS StdHandlers
{
public:
   StdHandlers();
   ~StdHandlers();
   
   void subscribe( Engine* engine );

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
   Class* functionClass() const { return m_functionClass; }

   /** Returns the global instance of the String class.
   \return the Engine instance of the String Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* stringClass() const { return m_stringClass; }

   /** Returns the global instance of the String class.
   \return the Engine instance of the String Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* rangeClass() const { return m_rangeClass; }

   /** Returns the global instance of the Array class.
   \return the Engine instance of the Array Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* arrayClass() const { return m_arrayClass; }

   /** Returns the global instance of the Dictionary class.
   \return the Engine instance of the Dictionary Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* dictClass() const { return m_dictClass; }

   /** Returns the global instance of the PseudoDictionary class.
   \return the Engine instance of the Dictionary Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* pseudoDictClass() const { return m_pseudoDictClass; }

   /** Returns the global instance of the Prototype class.
   \return the Engine instance of the Prototype Class (handler).

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* protoClass() const { return m_protoClass; }

   /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaClass() const { return m_metaClass; }

    /** Returns the global instance of the MetaFalconClass class.
   \return the Engine instance of the MetaFalconClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaFalconClass() const { return m_metaFalconClass; }

   /** Returns the global instance of the MetaHyperClass class.
   \return the Engine instance of the MetaHyperClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* metaHyperClass() const { return m_metaHyperClass; }

   /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* mantraClass() const { return m_mantraClass; }

      /** Returns the global instance of the MetaClass class.
   \return the Engine instance of the MetaClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* synFuncClass() const { return m_synFuncClass; }

   /** Returns the global instance of the GenericClass class.
   \return the Engine instance of the GenericClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* genericClass() const { return m_genericClass; }

   /** Returns the global instance of the ClassTreeStep class.
   \return the Engine instance of the ClassTreeStep handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* treeStepClass() const { return m_treeStepClass; }

   /** Returns the global instance of the statement class.
   \return the Engine instance of the ClassStatement handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* statementClass() const { return m_statementClass; }

   /** Returns the global instance of the ClassExpression class.
   \return the Engine instance of the ClassExpression handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* expressionClass() const { return m_exprClass; }

   /** Returns the global instance of the ClassSynTree class.
   \return the Engine instance of the ClassSynTree handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* syntreeClass() const { return m_syntreeClass; }

   /** Returns the global instance of the ClassSymbol class.
   \return the Engine instance of the ClassSymbol handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* symbolClass() const { return m_symbolClass; }

   /** Returns the global instance of the ClassStorer class.
   \return the Engine instance of the ClassStorer handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* storerClass() const { return m_storerClass; }

   /** Returns the global instance of the ClassRE class.
   \return the Engine instance of the ClassRE handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* reClass() const { return m_reClass; }

   /** Returns the global instance of the ClassRestorer class.
   \return the Engine instance of the ClassRestorer handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* restorerClass() const { return m_restorerClass; }

   /** Returns the global instance of the ClassStream class.
   \return the Engine instance of the ClassStream handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* streamClass() const { return m_streamClass; }

   /** Returns the global instance of the ClassStringStream class.
   \return the Engine instance of the ClassStringStream handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* stringStreamClass() const { return m_stringStreamClass; }

   /** Returns the global instance of the ClassClosure class.
   \return the Engine instance of the ClassClosure handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* closureClass() const { return m_closureClass; }

   /** Returns the global instance of the ClassClosedData class.
   \return the Engine instance of the ClassClosedData handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* closedDataClass() const { return m_closedDataClass; }

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
   ClassRawMem* rawMemClass() const { return m_rawMemClass; }

   /** Returns the global instance of the SharedClass class.
   \return the Engine instance of the SharedClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassShared* sharedClass() const { return m_sharedClass; }

   /** Returns the global instance of the ClassMessageQueue class.
   \return the Engine instance of the ClassMessageQueue handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* messageQueueClass() const { return m_messageQueueClass; }

   /** Returns the global instance of the ClassNumber class.
   \return the Engine instance of the ClassNumber handler.

    Method init() must have been called before.

    ClassNumber is the abstract base class for all kind of numbers,
    it's mainly to be used in select and other type checks.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* numberClass() const { return m_numberClass; }

   /** Returns the global instance of the FormatClass class.
   \return the Engine instance of the FormatClass handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* formatClass() const { return m_formatClass; }

   /** Returns the global instance of the ClassModule class.
   \return the Engine instance of the ClassModule handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   ClassModule* moduleClass() const { return m_moduleClass; }

   /** Returns the global instance of the ClassModSpace class.
   \return the Engine instance of the ClassModSpace handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* modSpaceClass() const { return m_modSpaceClass; }

   /** Returns the global instance of the ClassTimespace class.
   \return the Engine instance of the ClassTimespace handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* timestampClass() const { return m_timestampClass; }

   /** Returns the global instance of the ClassURI class.
   \return the Engine instance of the ClassURI handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* uriClass() const { return m_uriClass; }

   /** Returns the global instance of the ClassPath class.
   \return the Engine instance of the ClassPath handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* pathClass() const { return m_pathClass; }



   /** Returns the global instance of the ClassEventCourier class.
   \return the Engine instance of the ClassEventCourier handler.

    Method init() must have been called before.

    @note This method will assert and terminate the program if compiled in debug mode
    in case the engine has not been initialized. In release, it will just
    return a null pointer.
    */
   Class* eventCourierClass() const { return m_eventCourierClass; }

private:
   //===============================================
   // Global type handlers
   //
   Class* m_functionClass;
   Class* m_stringClass;
   Class* m_rangeClass;
   Class* m_arrayClass;
   Class* m_dictClass;
   Class* m_pseudoDictClass;
   Class* m_protoClass;
   Class* m_metaClass;
   Class* m_metaFalconClass;
   Class* m_metaHyperClass;
   Class* m_mantraClass;
   Class* m_synFuncClass;
   Class* m_genericClass;
   Class* m_formatClass;
   Class* m_numberClass;

   ClassRawMem* m_rawMemClass;
   ClassShared* m_sharedClass;

   //===============================================
   // Basic code reflection entities
   //
   Class* m_treeStepClass;
   Class* m_statementClass;
   Class* m_exprClass;
   Class* m_syntreeClass;
   Class* m_symbolClass;
   Class* m_closureClass;
   Class* m_closedDataClass;
   Class* m_storerClass;
   Class* m_reClass;
   Class* m_restorerClass;
   Class* m_streamClass;
   Class* m_stringStreamClass;
   ClassModule* m_moduleClass;
   Class* m_modSpaceClass;
   Class* m_messageQueueClass;
   Class* m_timestampClass;

   Class* m_uriClass;
   Class* m_pathClass;

   Class* m_eventCourierClass;
};

}

#endif

/* end of stdhandlers.h */

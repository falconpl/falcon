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

#undef SRC
#define SRC "engine/stdhandlers.cpp"

#include <falcon/stdhandlers.h>

//--- type headers ---
#include <falcon/classes/classfunction.h>
#include <falcon/classes/classsynfunc.h>
#include <falcon/classes/classcloseddata.h>
#include <falcon/classes/classclosure.h>
#include <falcon/classes/classcomposition.h>
#include <falcon/classes/classnumber.h>
#include <falcon/classes/classstring.h>
#include <falcon/classes/classrange.h>
#include <falcon/classes/classarray.h>
#include <falcon/classes/classdict.h>
#include <falcon/classes/classpseudodict.h>
#include <falcon/classes/classgeneric.h>
#include <falcon/classes/classformat.h>
#include <falcon/prototypeclass.h>
#include <falcon/classes/classnumeric.h>
#include <falcon/classes/classmantra.h>
#include <falcon/classes/classshared.h>
#include <falcon/classes/classmessagequeue.h>
#include <falcon/classes/metaclass.h>
#include <falcon/classes/metafalconclass.h>
#include <falcon/classes/metahyperclass.h>
#include <falcon/classes/classstorer.h>
#include <falcon/classes/classrestorer.h>
#include <falcon/classes/classstream.h>
#include <falcon/classes/classstringstream.h>
#include <falcon/classes/classre.h>
#include <falcon/classes/classtimestamp.h>

#include <falcon/classes/classuri.h>
#include <falcon/classes/classpath.h>

#include <falcon/classes/classmodule.h>
#include <falcon/classes/classmodspace.h>

#include <falcon/classes/classtreestep.h>
#include <falcon/classes/classstatement.h>
#include <falcon/classes/classexpression.h>
#include <falcon/classes/classsyntree.h>
#include <falcon/classes/classsymbol.h>
#include <falcon/classes/classrawmem.h>

#include <falcon/classes/classeventcourier.h>

namespace Falcon
{


StdHandlers::StdHandlers()
{
   // this won't be added to mantras,
   // so we'll create and delete it autonomously
   m_closedDataClass = new ClassClosedData;
}

void StdHandlers::subscribe(Engine* engine)
{

   ClassMantra* mantra = new ClassMantra;
   m_mantraClass = mantra;
   m_metaClass = new MetaClass;

   m_functionClass = new ClassFunction(mantra);
   m_stringClass = new ClassString;
   m_rangeClass = new ClassRange;
   m_arrayClass = new ClassArray;
   m_dictClass = new ClassDict;
   m_pseudoDictClass = new ClassPseudoDict;
   m_protoClass = new PrototypeClass;
   m_metaFalconClass = new MetaFalconClass;
   m_metaHyperClass = new MetaHyperClass;
   m_synFuncClass = new ClassSynFunc;
   m_genericClass = new ClassGeneric;
   m_formatClass = new ClassFormat;
   m_sharedClass = new ClassShared;
   m_numberClass = new ClassNumber;
   m_messageQueueClass = new ClassMessageQueue;

   // Notice: rawMem is not reflected, is used only in extensions.
   m_rawMemClass = new ClassRawMem();

   m_reClass = new ClassRE;
   m_closureClass = new ClassClosure;
   m_restorerClass = new ClassRestorer;
   m_storerClass = new ClassStorer;
   m_streamClass = new ClassStream;
   m_stringStreamClass = new ClassStringStream;

   m_symbolClass = new ClassSymbol;

   ClassTreeStep* ctreeStep = new ClassTreeStep;
   m_treeStepClass = ctreeStep;
   m_statementClass = new ClassStatement(ctreeStep);
   m_exprClass = new ClassExpression(ctreeStep);
   m_syntreeClass = new ClassSynTree(ctreeStep, static_cast<ClassSymbol*>(m_symbolClass));

   m_modSpaceClass = new ClassModSpace;
   m_moduleClass = new ClassModule;

   m_timestampClass = new ClassTimeStamp;

   m_uriClass = new ClassURI;
   m_pathClass = new ClassPath;

   m_eventCourierClass = new ClassEventCourier;

   engine->addMantra( m_functionClass );
   engine->addMantra( m_stringClass );
   engine->addMantra( m_arrayClass );
   engine->addMantra( m_dictClass );
   engine->addMantra( m_protoClass );
   engine->addMantra( m_metaClass );
   engine->addMantra( m_metaFalconClass );
   engine->addMantra( m_metaHyperClass );
   engine->addMantra( m_mantraClass );
   engine->addMantra( m_synFuncClass );
   engine->addMantra( m_genericClass );
   engine->addMantra( m_formatClass );
   engine->addMantra( m_rangeClass  );
   engine->addMantra( m_sharedClass );
   engine->addMantra( m_numberClass );
   engine->addMantra( m_messageQueueClass );

   engine->addMantra(m_closureClass);
   engine->addMantra(m_treeStepClass);
   engine->addMantra(m_statementClass);
   engine->addMantra(m_exprClass);
   engine->addMantra(m_syntreeClass);
   engine->addMantra(m_symbolClass);
   engine->addMantra(m_modSpaceClass);
   engine->addMantra(m_moduleClass);
   engine->addMantra(m_restorerClass);
   engine->addMantra(m_storerClass);
   engine->addMantra(m_reClass);
   engine->addMantra(m_streamClass);
   engine->addMantra(m_stringStreamClass);
   engine->addMantra(m_timestampClass);

   engine->addMantra(m_uriClass);
   engine->addMantra(m_pathClass);

   engine->addMantra(m_eventCourierClass);

   engine->addMantra( new ClassComposition );
}

StdHandlers::~StdHandlers()
{
   // delete unregistered handlers
   delete m_closedDataClass;
}

}

/* end of stdhandlers.cpp */

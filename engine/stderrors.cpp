/*
   FALCON - The Falcon Programming Language
   FILE: stderrors.cpp

   Engine static/global data setup and initialization
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 31 Jul 2011 15:30:08 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "engine/stderrors.cpp"
#include <falcon/error.h>
#include <falcon/stderrors.h>
#include <falcon/errorclasses.h>

namespace Falcon
{

//============================================================
// The main class.
//

StdErrors::StdErrors()
{
   m_error = new ClassError("Error");
   m_code = new ClassCodeError;
   m_generic = new ClassGenericError;
   m_operand = new ClassOperandError;
   m_unsupported = new ClassUnsupportedError;
   m_io = new ClassIOError;
   m_interrupted = new ClassInterruptedError;
   m_encoding = new ClassEncodingError;
   m_access = new ClassAccessError;
   m_accessType = new ClassAccessTypeError;
   m_syntax = new ClassSyntaxError;
   m_param =  new ClassParamError;
   m_math = new ClassMathError;
   m_link = new ClassLinkError;
   m_unserializable = new ClassUnserializableError;
   m_type = new ClassTypeError;
}


StdErrors::~StdErrors()
{  
}

void StdErrors::addBuiltins() const
{
   static Engine* eng = Engine::instance();
   
   eng->addMantra(m_error);
   eng->addMantra(m_code);
   eng->addMantra(m_generic);
   eng->addMantra(m_operand);
   eng->addMantra(m_unsupported);
   eng->addMantra(m_io);
   eng->addMantra(m_interrupted);
   eng->addMantra(m_encoding);
   eng->addMantra(m_access);
   eng->addMantra(m_accessType);
   eng->addMantra(m_syntax);
   eng->addMantra(m_param);
   eng->addMantra(m_link);
   eng->addMantra(m_unserializable);
   eng->addMantra(m_type);
}
   
}

/* end of stderrors.cpp */

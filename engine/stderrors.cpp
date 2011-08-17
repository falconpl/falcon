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
   m_link = new ClassLinkError;
}


StdErrors::~StdErrors()
{
   delete m_error;
   delete m_code;
   delete m_generic;
   delete m_operand;
   delete m_unsupported;
   delete m_io;
   delete m_interrupted;
   delete m_encoding;
   delete m_access;
   delete m_accessType;
   delete m_syntax;
   delete m_param;
   delete m_link;
}

void StdErrors::addBuiltins() const
{
   static Engine* eng = Engine::instance();
   
   eng->addBuiltin(m_error);
   eng->addBuiltin(m_code);
   eng->addBuiltin(m_generic);
   eng->addBuiltin(m_operand);
   eng->addBuiltin(m_unsupported);
   eng->addBuiltin(m_io);
   eng->addBuiltin(m_interrupted);
   eng->addBuiltin(m_encoding);
   eng->addBuiltin(m_access);
   eng->addBuiltin(m_accessType);
   eng->addBuiltin(m_syntax);
   eng->addBuiltin(m_param);
   eng->addBuiltin(m_link);
}
   
}

/* end of stderrors.cpp */

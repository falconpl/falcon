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
   m_code = new CodeErrorClass;
   m_generic = new GenericErrorClass;
   m_operand = new OperandErrorClass;
   m_unsupported = new UnsupportedErrorClass;
   m_io = new IOErrorClass;
   m_interrupted = new InterruptedErrorClass;
   m_encoding = new EncodingErrorClass;
   m_access = new AccessErrorClass;
   m_accessType = new AccessTypeErrorClass;
   m_syntax = new SyntaxErrorClass;
   m_param =  new ParamErrorClass;
   m_link = new LinkErrorClass;
}


StdErrors::~StdErrors()
{
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
   
}

/* end of stderrors.cpp */

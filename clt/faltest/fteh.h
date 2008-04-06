/*
   FALCON - The Falcon Programming Language.
   FILE: fteh.h

   ErrorHandler for the faltest application.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer feb 15 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   ErrorHandler for the faltest application.
*/

#ifndef flc_FTErrorHandler_H
#define flc_FTErrorHandler_H

#include <falcon/setup.h>
#include <falcon/errhand.h>
#include <falcon/string.h>

namespace Falcon {

/** The flctest ErrorHandler.
   This class is meant to record the error messages and then
   provide all of them to the embedding application.

   This is more or less the minimal error handler that any
   embedding application should have.

   \note This error handler uses VMUtils to provide an error
   reporting consistent with falcon command line. Embedding
   applications are not required to do this.
*/
class FTErrorHandler: public ErrorHandler
{
   String m_error;

public:

   FTErrorHandler() {}
   virtual void handleError( Error *preformatted );
   const String &getError( ) const { return m_error; }
};

}

#endif

/* end of fteh.h */

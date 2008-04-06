/*
   FALCON - The Falcon Programming Language.
   FILE: fteh.cpp

   ErrorHandler for the faltest application - implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer feb 15 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   ErrorHandler for the faltest application - implementation.
*/

#include "fteh.h"
#include <falcon/string.h>

namespace Falcon {

void FTErrorHandler::handleError( Error *preformatted )
{
   m_error += preformatted->toString() + "\n";
}

}

/* end of fteh.cpp */

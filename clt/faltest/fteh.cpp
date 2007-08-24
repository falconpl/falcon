/*
   FALCON - The Falcon Programming Language.
   FILE: fteh.cpp
   $Id: fteh.cpp,v 1.4 2007/03/04 17:39:03 jonnymind Exp $

   ErrorHandler for the faltest application - implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mer feb 15 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

/*
   FALCON - The Falcon Programming Language
   FILE: stackframe.cpp
   $Id: stackframe.cpp,v 1.1 2006/10/15 20:21:50 gian Exp $

   Implementation for stack frame functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 15 2006
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
   Implementation for stack frame functions
*/

#include <falcon/stackframe.h>

namespace Falcon {
void StackFrame_deletor( void *data )
{
   StackFrame *sf = (StackFrame *) data;
   delete sf;
}

}


/* end of stackframe.cpp */

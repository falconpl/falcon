/*
   FALCON - The Falcon Programming Language
   FILE: stackframe.cpp

   Implementation for stack frame functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 15 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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

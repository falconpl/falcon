/*
   FALCON - The Falcon Programming Language.
   FILE: fatal.h

   Function signaling a fatal error (which should abort).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Mar 2011 16:44:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_FATAL_H_
#define _FALCON_FATAL_H_

#include <falcon/setup.h>

namespace Falcon
{

/** Terminates the host process.

 This function pointer is normally set to a function terminating the host
 process with an error message sent to output through vsprintf.

 After the message is printed, abort() is called.

 The host can redirect it before initializing the engine.
*/
extern FALCON_DYN_SYM void (*fatal)( const char* msg, ... );

}

#endif	/* _FALCON_FATAL_H_ */

/* end of fatal.h */

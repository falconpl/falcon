/*
   FALCON - The Falcon Programming Language.
   FILE: interruptederror.h

   Class representing an error thrown when a wait is interrupted.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Mar 2011 13:27:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_INTERRUPTEDERROR_H
#define	FALCON_INTERRUPTEDERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Class representing an error thrown when a wait is interrupted. */
class FALCON_DYN_CLASS InterruptedError: public Error
{
public:
   InterruptedError( const ErrorParam &params );

protected:
   virtual ~InterruptedError();
};

}

#endif	/* FALCON_INTERRUPTEDERROR_H */

/* end of interruptederror.h */

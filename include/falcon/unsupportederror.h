/*
   FALCON - The Falcon Programming Language.
   FILE: unsupportederror.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Mar 2011 13:27:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_UNSUPPORTEDERROR_H
#define	FALCON_UNSUPPORTEDERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error thrown when using a feature exposed but not supported by the target object. */
class FALCON_DYN_CLASS UnsupportedError: public Error
{
public:
   UnsupportedError( const ErrorParam &params );

protected:
   virtual ~UnsupportedError();
};

}

#endif	/* UNSUPPORTEDERROR_H */

/* end of unsupportederror.h */

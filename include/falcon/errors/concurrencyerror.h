/*
   FALCON - The Falcon Programming Language.
   FILE: concurrenceerror.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 11 Apr 2013 16:17:40 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CONCURRENCYERROR_H
#define FALCON_CONCURRENCYERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error thrown when using a feature exposed but not supported by the target object. */
class FALCON_DYN_CLASS ConcurrencyError: public Error
{
public:
   ConcurrencyError( );
   ConcurrencyError( const ErrorParam &params );

protected:
   virtual ~ConcurrencyError();
};

}

#endif

/* end of concurrenceerror.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: paramerror.h

   Error class raised when functions receive invalid parameters.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 19:28:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PARAMERROR_H
#define	FALCON_PARAMERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error thrown when functions receive invalid parameters. */
class FALCON_DYN_CLASS ParamError: public Error
{
public:
   ParamError( const ErrorParam &params );

protected:
   virtual ~ParamError();
};

}

#endif	/* FALCON_PARAMERROR_H */

/* end of paramerror.h */

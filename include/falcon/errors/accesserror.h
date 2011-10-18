/*
   FALCON - The Falcon Programming Language.
   FILE: accesserror.h

   Error while accessing objects (with square or dot operator).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 02 Jun 2011 20:59:01 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ACCESSERROR_H
#define FALCON_ACCESSERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error while accessing objects (with square or dot operator).*/
class FALCON_DYN_CLASS AccessError: public Error
{
public:
   AccessError( const ErrorParam &params );

protected:
   virtual ~AccessError();
};

}

#endif	/* FALCON_ACCESSERROR_H */

/* end of accesserror.h */

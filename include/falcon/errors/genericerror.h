/*
   FALCON - The Falcon Programming Language.
   FILE: genericerror.h

   Generic Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_GENERICERROR_H
#define FALCON_GENERICERROR_H

#include <falcon/error.h>

namespace Falcon {

class FALCON_DYN_CLASS GenericError: public Error
{
public:
   GenericError( const ErrorParam &params );

protected:
   virtual ~GenericError();
};

}

#endif	/* FALCON_GENERICERROR_H */

/* end of genericerror.h */

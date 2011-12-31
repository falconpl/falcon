/*
   FALCON - The Falcon Programming Language.
   FILE: codeerror.h

   Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CODEERROR_H
#define FALCON_CODEERROR_H

#include <falcon/error.h>

namespace Falcon {

class FALCON_DYN_CLASS CodeError: public Error
{
public:
   CodeError( const ErrorParam &params );

protected:
   virtual ~CodeError();
};

}

#endif	/* FALCON_CODEERROR_H */

/* end of codeerror.h */

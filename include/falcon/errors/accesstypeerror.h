/*
   FALCON - The Falcon Programming Language.
   FILE: accesstypeerror.h

   Error while accessing objects in read/write mode.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 19 Jun 2011 15:56:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ACCESSTYPEERROR_H
#define	FALCON_ACCESSTYPEERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error while accessing objects read/write mode.*/
class FALCON_DYN_CLASS AccessTypeError: public Error
{
public:
   AccessTypeError( );
   AccessTypeError( const ErrorParam &params );

protected:
   virtual ~AccessTypeError();
};

}

#endif	/* FALCON_ACCESSTYPEERROR_H */

/* end of accesstypeerror.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: ioerror.h

   I/O Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 12 Mar 2011 13:27:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_IOERROR_H
#define	FALCON_IOERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error thrown when an I/O Error is found.
 */
class FALCON_DYN_CLASS IOError: public Error
{
public:
   IOError( const ErrorParam &params );

protected:
   virtual ~IOError();
};

}

#endif	

/* end of ioerror.h */

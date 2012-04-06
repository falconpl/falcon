/*
   FALCON - The Falcon Programming Language.
   FILE: encodingerror.h

   Generic Error class.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Mar 2011 13:04:14 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ENCODINGERROR_H
#define	FALCON_ENCODINGERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Class being raised when detecting an error in text encodings.*/
class FALCON_DYN_CLASS EncodingError: public Error
{
public:
   EncodingError( );
   EncodingError( const ErrorParam &params );

protected:
   virtual ~EncodingError();
};

}

#endif	/* FALCON_ENCODINGERROR_H */

/* end of encodingerror.h */

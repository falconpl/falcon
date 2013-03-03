/*
   FALCON - The Falcon Programming Language.
   FILE: matherror.h

   Error class raised when invalid mathematical operations are performed.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai & Paul Davey
   Begin: Sat, 07 May 2011 19:28:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_MATHERROR_H
#define	FALCON_MATHERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error thrown when invalid mathematical operations are performed. */
class FALCON_DYN_CLASS MathError: public Error
{
public:
   MathError( );
   MathError( const ErrorParam &params );

protected:
   virtual ~MathError();
};

}

#endif	/* FALCON_MATHERROR_H */

/* end of matherror.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: syntaxerror.h

   Error class representing compilation errors.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 17 Apr 2011 20:13:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_SYNTAXERROR_H
#define	FALCON_SYNTAXERROR_H

#include <falcon/error.h>

namespace Falcon {

class FALCON_DYN_CLASS SyntaxError: public Error
{
public:
   SyntaxError( const ErrorParam &params );

protected:
   virtual ~SyntaxError();
};

}

#endif	/* FALCON_SYNTAXERROR_H */

/* end of syntaxerror.h */

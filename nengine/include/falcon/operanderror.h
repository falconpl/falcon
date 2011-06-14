/*
   FALCON - The Falcon Programming Language.
   FILE: operanderror.h

   Error class: when applying operand to items not supporting them.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_OPERANDERROR_H
#define	FALCON_OPERANDERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error raised when applying operand to items not supporting them.
 */

class FALCON_DYN_CLASS OperandError: public Error
{
public:
   OperandError( const ErrorParam &params );

protected:
   virtual ~OperandError();
};

}

#endif	/* FALCON_OPERANDERROR_H */

/* end of operanderror.h */

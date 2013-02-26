/*
   FALCON - The Falcon Programming Language.
   FILE: unserializableerror.h

   Error class: when trying to serialize an object that can't be ser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 26 Feb 2013 14:26:15 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_TYPEERROR_H
#define FALCON_TYPEERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error raised when an expected type is not found
 */

class FALCON_DYN_CLASS TypeError: public Error
{
public:
   TypeError( );
   TypeError( const ErrorParam &params );

protected:
   virtual ~TypeError();
};

}

#endif	/* FALCON_UNSERIALIZABLEERROR_H */

/* end of typeerror.h */

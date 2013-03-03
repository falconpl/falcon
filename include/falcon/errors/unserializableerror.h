/*
   FALCON - The Falcon Programming Language.
   FILE: unserializableerror.h

   Error class: when trying to serialize an object that can't be ser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_UNSERIALIZABLEERROR_H
#define FALCON_UNSERIALIZABLEERROR_H

#include <falcon/error.h>

namespace Falcon {

/** Error raised when when trying to serialize an object that can't be serialized.
 */

class FALCON_DYN_CLASS UnserializableError: public Error
{
public:
   UnserializableError( );
   UnserializableError( const ErrorParam &params );

protected:
   virtual ~UnserializableError();
};

}

#endif	/* FALCON_UNSERIALIZABLEERROR_H */

/* end of unserializableerror.h */

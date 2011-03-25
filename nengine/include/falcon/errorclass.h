/*
   FALCON - The Falcon Programming Language.
   FILE: error.h

   Class for storing error in scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_ERRORCLASS_H
#define	FALCON_ERRORCLASS_H

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/class.h>

namespace Falcon {

class Stream;

/** The base class of all the error class hierarcy.
 All the hierarcy expects an ErrorParam instance as the
 creationParameter for the create() method.
 
 */
class FALCON_DYN_SYM ErrorClass: public Class
{
public:
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   // todo: overload properties.

protected:
   ErrorClass( const String& name );
   virtual ~ErrorClass();

};

}

#endif	/* FALCON_ERROR_H */

/* end of errorclass.h */

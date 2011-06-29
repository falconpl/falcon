/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.h

   Handler for classes defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 16:04:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CORECLASS_H_
#define _FALCON_CORECLASS_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/** Handler for classes defined by a Falcon script.

 This class implements a class handler for classes a Falcon script. In other words,
 it is a handler for the "class type". The content of this type is a FalconClass,
 where properties and methods declared in a Falcon script class declaration
 are stored.
 
 */
class FALCON_DYN_CLASS CoreClass: public Class
{
public:

   CoreClass();
   virtual ~CoreClass();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   //=============================================================

   // virtual void op_create( VMachine *vm, int32 pcount ) const; -- let the default non-creable thing to work
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;
   virtual void op_call( VMachine *vm, int32 pcount, void* self ) const;
};

}

#endif /* _FALCON_CORECLASS_H_ */

/* end of coreclass.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: coreclass.h

   Class defined by a Falcon script.
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

/** Class defined by a Falcon script.

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

   bool isCoreClass();

   virtual void* create( void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target ) const;

   //=============================================================

   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_call( VMachine *vm, int32 pcount, void* self ) const;
};

}

#endif /* _FALCON_CORECLASS_H_ */

/* end of coreclass.h */

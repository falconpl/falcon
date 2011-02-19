/*
   FALCON - The Falcon Programming Language.
   FILE: coreint.h

   Nil object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREINT_H_
#define _FALCON_COREINT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/**
 Class handling a string as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreInt: public Class
{
public:

   CoreInt();
   virtual ~CoreInt();

   virtual void* create( void* creationParams ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( Stream* stream, void* self ) const;
   virtual void* deserialize( Stream* stream ) const;

   virtual void describe( void* instance, String& target ) const;

   //=============================================================

   virtual void op_isTrue( VMachine *vm, void* self, Item& target ) const;
};

}

#endif /* _FALCON_CORENIL_H_ */

/* end of coreint.h */

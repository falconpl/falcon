/*
   FALCON - The Falcon Programming Language.
   FILE: corebool.h

   Class defined by a Falcon script.
   -------------------------------------------------------------------
   Author: Francesco Magliocca
   Begin: Sun, 19 Jun 2011 12:40:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_COREBOOL_H_
#define _FALCON_COREBOOL_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon
{

/** Class defined by a Falcon script.

 This class implements a class handler for classes a Falcon script. In other words,
 it is a handler for the "class type". The content of this type is a FalconClass,
 where properties and methods declared in a Falcon script class declaration
 are stored.
 
 */
class FALCON_DYN_CLASS CoreBool : public Class
{
public:

   CoreBool();
   virtual ~CoreBool();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   //=============================================================

   virtual void op_create( VMachine *vm, int32 pcount ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;

private:

   class FALCON_DYN_CLASS NextOpCreate: public PStep {
   public:
      NextOpCreate() { apply = apply_; }
      static void apply_( const PStep*, VMachine* vm );
   } m_OP_create_next;
};

}

#endif /* _FALCON_COREBOOL_H_ */

/* end of coreclass.h */
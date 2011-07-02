/*
   FALCON - The Falcon Programming Language.
   FILE: classstring.h

   String object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSTRING_H_
#define _FALCON_CLASSSTRING_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/string.h>

#include <falcon/pstep.h>
namespace Falcon
{

/**
 Class handling a string as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassString: public Class
{
public:

   ClassString();
   virtual ~ClassString();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   //=============================================================
   virtual void op_create( VMachine *vm, int32 pcount ) const;

   virtual void op_add( VMachine *vm, void* self ) const;
   virtual void op_aadd( VMachine *vm, void* self ) const;

   // THIS IS A TODO!
   virtual void op_getIndex( VMachine *vm, void* self ) const;
   
   virtual void op_compare( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;
   virtual void op_true( VMachine *vm, void* self ) const;

private:

   class FALCON_DYN_CLASS NextOp: public PStep {
   public:
      NextOp();
      static void apply_( const PStep*, VMachine* vm );
   } m_nextOp;
};

}

#endif /* _FALCON_CLASSSTRING_H_ */

/* end of classstring.h */

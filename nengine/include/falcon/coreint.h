/*
   FALCON - The Falcon Programming Language.
   FILE: coreint.h

   Int object handler.
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
 Class handling an int as an item in a falcon script.
 */

class FALCON_DYN_CLASS CoreInt: public Class
{
public:

   CoreInt();
   virtual ~CoreInt();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   //=============================================================

   virtual void op_create( VMachine *vm, int32 pcount ) const;
   virtual void op_isTrue( VMachine *vm, void* self ) const;
   virtual void op_toString( VMachine *vm, void* self ) const;

   virtual void op_add( VMachine *vm, void* self ) const;

   virtual void op_sub( VMachine *vm, void* self ) const;

   virtual void op_mul( VMachine *vm, void* self ) const;

   virtual void op_div( VMachine *vm, void* self ) const;

   virtual void op_mod( VMachine *vm, void* self ) const;

   virtual void op_pow( VMachine *vm, void* self ) const;

   virtual void op_aadd( VMachine *vm, void* self) const;

   virtual void op_asub( VMachine *vm, void* self ) const;

   virtual void op_amul( VMachine *vm, void* self ) const;

   virtual void op_adiv( VMachine *vm, void* self ) const;

   virtual void op_amod( VMachine *vm, void* self ) const;

   virtual void op_apow( VMachine *vm, void* self ) const;

   virtual void op_inc(VMachine *vm, void* self ) const;

   virtual void op_dec(VMachine *vm, void* self) const;

   virtual void op_incpost(VMachine *vm, void* self ) const;

   virtual void op_decpost(VMachine *vm, void* self ) const;
};

}

#endif /* _FALCON_COREINT_H_ */

/* end of coreint.h */

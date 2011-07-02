/*
   FALCON - The Falcon Programming Language.
   FILE: classint.h

   Int object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 13 Feb 2011 15:11:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSINT_H_
#define _FALCON_CLASSINT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

/**
 Class handling an int as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassInt: public Class
{
public:

   ClassInt();
   virtual ~ClassInt();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;

   //=============================================================

   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_mul( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* self ) const;
   virtual void op_mod( VMContext* ctx, void* self ) const;
   virtual void op_pow( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_amul( VMContext* ctx, void* self ) const;
   virtual void op_adiv( VMContext* ctx, void* self ) const;
   virtual void op_amod( VMContext* ctx, void* self ) const;
   virtual void op_apow( VMContext* ctx, void* self ) const;
   virtual void op_inc( VMContext* ctx, void* self ) const;
   virtual void op_dec(VMContext* ctx, void* self) const;
   virtual void op_incpost(VMContext* ctx, void* self ) const;
   virtual void op_decpost(VMContext* ctx, void* self ) const;
};

}

#endif /* _FALCON_CLASSINT_H_ */

/* end of classint.h */

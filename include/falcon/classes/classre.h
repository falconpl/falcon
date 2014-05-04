/*
   FALCON - The Falcon Programming Language.
   FILE: classre.h

   RE2 object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 03 Feb 2013 13:49:49 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSRE_H_
#define _FALCON_CLASSRE_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/string.h>

#include <falcon/pstep.h>
namespace Falcon
{

class FALCON_DYN_CLASS ClassRE: public Class
{
public:

   ClassRE();
   virtual ~ClassRE();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext*, DataWriter* dw, void* data ) const;
   virtual void restore( VMContext* , DataReader* dr ) const;

   virtual void describe( void* instance, String& target, int, int ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   //=============================================================
   virtual bool op_init( VMContext* ctx, void*, int32 pcount ) const;

   virtual void op_compare( VMContext* ctx, void* self ) const;
   virtual void op_div( VMContext* ctx, void* instance ) const;
   virtual void op_mod( VMContext* ctx, void* instance ) const;
   virtual void op_mul( VMContext* ctx, void* instance ) const;
   virtual void op_pow( VMContext* ctx, void* instance ) const;

};

}

#endif /* _FALCON_CLASSRE_H_ */

/* end of classre.h */

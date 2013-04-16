/*
   FALCON - The Falcon Programming Language.
   FILE: classdict.h

   Standard language dictionary object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 27 Apr 2011 14:15:37 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSDICT_H_
#define _FALCON_CLASSDICT_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a dictionary as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassDict: public Class
{
public:

   ClassDict();
   virtual ~ClassDict();

   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMarkInstance( void* self, uint32 mark ) const;

   //=============================================================

   virtual bool op_init( VMContext* ctx, void* inst, int32 pcount ) const;

   virtual void op_add( VMContext* ctx, void* self ) const;
   virtual void op_aadd( VMContext* ctx, void* self ) const;
   virtual void op_sub( VMContext* ctx, void* self ) const;
   virtual void op_asub( VMContext* ctx, void* self ) const;
   virtual void op_shl( VMContext* ctx, void* self ) const;
   virtual void op_ashl( VMContext* ctx, void* self ) const;
   virtual void op_shr( VMContext* ctx, void* self ) const;
   virtual void op_ashr( VMContext* ctx, void* self ) const;

   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_in( VMContext* ctx, void* instance ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

protected:
   ClassDict( const String& subclsName );

private:
   void init();
};

}

#endif /* _FALCON_CLASSDICT_H_ */

/* end of classdict.h */

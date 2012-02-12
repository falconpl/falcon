/*
   FALCON - The Falcon Programming Language.
   FILE: classrange.h

   Standard language range object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai & Paul Davey
   Begin: Mon, 25 Jul 2011 23:04 +1200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSRANGE_H_
#define _FALCON_CLASSRANGE_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a range as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassRange: public Class
{
public:

   ClassRange();
   virtual ~ClassRange();

   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance( void* source ) const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual bool gcCheck( void* self, uint32 mark ) const;
   virtual void enumerateProperties( void* self, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* self, PVEnumerator& cb ) const;

   //=============================================================

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;

   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_getIndex( VMContext* ctx, void* self ) const;
   virtual void op_setIndex( VMContext* ctx, void* self ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

private:
};

}

#endif /* _FALCON_CLASSRANGE_H_ */

/* end of classrange.h */

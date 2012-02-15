/*
   FALCON - The Falcon Programming Language.
   FILE: classclosure.h

   Handler for closure entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Jan 2012 16:13:09 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSCLOSURE_H_
#define _FALCON_CLASSCLOSURE_H_

#include <falcon/setup.h>
#include <falcon/class.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**  Class handling an array as an item in a falcon script.
 
 */

class FALCON_DYN_CLASS ClassClosure: public Class
{
public:

   ClassClosure();
   virtual ~ClassClosure();
   
   //=============================================================

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual bool gcCheck( void* instance, uint32 mark ) const;
   
   //=============================================================

   virtual void op_call( VMContext* ctx, int32 paramCount, void* instance ) const;
};

}

#endif /* _FALCON_CLASSARRAY_H_ */

/* end of classclosure.h */

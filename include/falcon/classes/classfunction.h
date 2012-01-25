/*
   FALCON - The Falcon Programming Language.
   FILE: classfunction.h

   Function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSFUNCTION_H_
#define FALCON_CLASSFUNCTION_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon
{

class Function;
class Module;

/**
 Class handling a function as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassFunction: public Class
{
public:

   ClassFunction();
   virtual ~ClassFunction();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   
   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream, void*& empty ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMark( void* self, uint32 mark ) const;
   virtual bool gcCheck( void* self, uint32 mark ) const;

   //=====================================================
   // Operators.
   //
   // Can a function instance be created?
   //virtual void op_create( VMContext* ctx, int32 pcount ) const;
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_eval( VMContext* ctx, void* self ) const;

};

}

#endif /* FUNCTION_H_ */

/* end of classfunction.h */

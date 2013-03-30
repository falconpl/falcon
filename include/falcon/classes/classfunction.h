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
#include <falcon/classes/classmantra.h>


namespace Falcon
{

class Function;
class Module;

/**
 Class handling a function as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassFunction: public ClassMantra
{
public:

   ClassFunction(ClassMantra* parent);
   virtual ~ClassFunction();
   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;
   virtual void* createInstance() const;

   //=====================================================
   // Operators.
   //
   // Can a function instance be created?
   virtual void op_call( VMContext* ctx, int32 paramCount, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

protected:
   ClassFunction( const String& name, int64 type ):
      ClassMantra( name, type )
   {}
};

}

#endif /* FUNCTION_H_ */

/* end of classfunction.h */

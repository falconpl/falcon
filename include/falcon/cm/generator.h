/*
   FALCON - The Falcon Programming Language.
   FILE: generator.h

   Falcon core module -- Generator class to help iterating
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Mar 2013 17:59:00 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_GENERATOR_H_
#define _FALCON_GENERATOR_H_

#include <falcon/setup.h>
#include <falcon/fassert.h>
#include <falcon/class.h>
#include <falcon/pstep.h>

namespace Falcon {
class Shared;

namespace Ext {

/*#
   @class Generator
   @brief Helper for iterations on complex object.
   @param func Generator function or callable entity
   @param data The data on which the loop is performed.
   @optparam iterator Initial value for the iterator property.
   @ingroup parallel

   Instances of this class override the internal op_next VM operand so that
   each next request (in rules, for/in loop, functional loops as @a map() and the like)
   so that a generator so created:

   @code
     gen = Generator( func, data, iterator )
   @endcode

   is resolved through the invocation of the following code:

   @code
    func( data, gen )
   @endcode

   The function @b func can then:
   - return break to terminate any loop immediately
   - return a simple value to yield the last value of the iteration (used in @b forlast).
   - return with doubt (@b return? statement) to declare more values awaiting after this one.

   The generator instance keeps a @a Generator.iterator property that is used
   for the function @b func to memorize the status of the iteration. The value is initialized
   to nil, unless a @b iterator parameter is given at generator creation.

   The Generator class has also a @b hasNext and @b next property set that can be used in
   unstructure loops (like i.e. while). For instance:

   @code
     function iter( data, gen )
         value = data[gen.iterator++]
         if gen.iterator == data.len
            // the last one:
            return value
         else
            return? value  // more to come..
         end
     end

     gen = Generator( func, .[1 2 3 4], 0 )

     while gen.hasNext
        > "Next... ", gen.next
     end

     >"This is equivalent to... "

     for value in Generator( func, .[1 2 3 4], 0 )
        > "Next... ", value
     end
   @endcode

   @prop next Yields the next value returned by the given iterator function
   @prop hasNext true if the iterator function is expected to return other values
   @porp iterator An arbitrary item used by the iterator function to iterate.
   @porp func The generator function (read only)
   @porp data The generator data (read only)
*/
class ClassGenerator: public Class
{
public:
   ClassGenerator();
   virtual ~ClassGenerator();

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;

   void store( VMContext*, DataWriter*, void* ) const;
   void restore( VMContext*, DataReader*) const;
   void flatten( VMContext*, ItemArray&, void* ) const;
   void unflatten( VMContext*, ItemArray&, void* ) const;

   //=============================================================
   //
   virtual bool op_init( VMContext* ctx, void* instance, int pcount ) const;

   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop ) const;

   virtual void op_iter( VMContext* ctx, void* instance ) const;
   virtual void op_next( VMContext* ctx, void* instance ) const;

private:

   FALCON_DECLARE_INTERNAL_PSTEP(AfterHasNext);
   FALCON_DECLARE_INTERNAL_PSTEP(AfterNext);
};

}
}


#endif

/* end of waiter.h */

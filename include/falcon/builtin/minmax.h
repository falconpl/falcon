/*
   FALCON - The Falcon Programming Language.
   FILE: minmax.h

   Definitions for min() and max() functions and pseudo-functions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 30 Apr 2011 19:05:23 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_MINMAX_H_
#define _FALCON_MINMAX_H_

#include <falcon/pseudofunc.h>
#include <falcon/string.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS MinOrMax: public PseudoFunction
{
public:
   MinOrMax( const String& name, bool bIsMax );
   virtual ~MinOrMax();
   virtual void invoke( VMContext* vm, int32 pCount = 0 );

   //Need to do something about this
   bool m_bIsMax;

private:

   // Step for invocation
   class FALCON_DYN_CLASS InvokeStep: public PStep
   {
   public:
      InvokeStep( bool isMax );
      virtual ~InvokeStep() {}
      static void apply_( const PStep* ps, VMContext* vm );

      bool m_bIsMax;
      // Step if the first item has a compare overload
      class FALCON_DYN_CLASS CompareStep: public PStep
      {
      public:
         CompareStep( bool isMax );
         virtual ~CompareStep() {}
         static void apply_( const PStep* ps, VMContext* vm );

         bool m_bIsMax;
      };

      CompareStep m_compare;
   };


   // Step if the first item has a compare overload
   class FALCON_DYN_CLASS CompareNextStep: public PStep
   {
   public:
      CompareNextStep( bool isMax );
      virtual ~CompareNextStep() {}
      static void apply_( const PStep* ps, VMContext* vm );

      bool m_bIsMax;
   };

   InvokeStep m_invoke;
   CompareNextStep m_compareNext;
};


/*# @function max
   @param first First operand
   @param second Second operand
   @return The the highest value between first and second
   @inset builtin

 This function is defined as:
 @code
 function max(first, second)
   return (first >= second) ? first : second
 end
 @endcode

 If the first operand is an object with the __compare method overloaded,
 that will be invoked to determine the comparation order.

 If the items are considered equal or having the same ordering, the first
 item is returned.
*/

/** Max pseudofunction. */
class FALCON_DYN_CLASS Max: public MinOrMax
{
public:
   Max();
   virtual ~Max();
};

/*# @function min
   @param first First operand
   @param second Second operand
   @return The the lowest value between first and second
   @inset builtin

 This function is defined as:
 @code
 function max(first, second)
   return (first >= second) ? first : second
 end
 @endcode

 If the first operand is an object with the __compare method overloaded,
 that will be invoked to determine the comparation order.
*/
class FALCON_DYN_CLASS Min: public MinOrMax
{
public:
   Min();
   virtual ~Min();
};

}
}

#endif

/* end of minmax.h */

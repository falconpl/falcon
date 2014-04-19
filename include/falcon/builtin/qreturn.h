/*
   FALCON - The Falcon Programming Language.
   FILE: qreturn.h

   Falcon core module -- qreturn function
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 19 Apr 2014 13:36:35 +0200

   -------------------------------------------------------------------
   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_QRETURN_H
#define FALCON_CORE_QRETURN_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function returnq
   @brief Conditionally return from the current context.
   @param check a value that can be true or false
   @param value The value to be returned.
   @return Never returns.

   This function exits the parent function, retunrning either a deterministic or
   non-deterministic (doubtful) value, depending on the @b check paramter.

   The following pseudocodes are equivalent:

   @code
   if someCond
      reutrn? value
   else
      return value
   end

   // equivalent to
   returnq( someCond, value )
   @endcode

   If the @b check paramter is true, the caller is notified that further calls
   to the funciton using @b qreturn might return a different value.

   If it's false, then the caller is informed that the given value is definitive,
   or that is the last value in a sequence.
*/

class FALCON_DYN_CLASS QReturn: public PseudoFunction
{
public:
   QReturn();
   virtual ~QReturn();
   virtual void invoke( VMContext* vm, int32 nParams );

private:
   
   class FALCON_DYN_CLASS Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      virtual ~Invoke() {}
      static void apply_( const PStep* ps, VMContext* vm );

   };

   Invoke m_invoke;
};

}
}

#endif   /* FALCON_CORE_QRETURN_H */

/* end of qreturn.h */

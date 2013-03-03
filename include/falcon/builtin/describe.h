/*
   FALCON - The Falcon Programming Language.
   FILE: describe.h

   Falcon core module -- describe function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_DESCRIBE_H
#define FALCON_CORE_DESCRIBE_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function describe
   @brief Returns a basic description of the object.
   @param item an item of any kind
   @return A string representing the contents of an object.

*/

/*#
   @method describe BOM

   @function describe
   @brief Returns a basic description of the object.
   @return A string representing the contents of an object.
*/

class FALCON_DYN_CLASS Describe: public PseudoFunction
{
public:
   Describe();
   virtual ~Describe();
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

#endif

/* end of describe.h */

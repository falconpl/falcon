/*
   FALCON - The Falcon Programming Language.
   FILE: baseclass.h

   Falcon core module -- Returns the class of an item
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 28 Dec 2011 10:54:08 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_BASECLASS_H
#define FALCON_CORE_BASECLASS_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function baseClass
   @brief Gets the class of an item
   @param item an item of any kind
   @return a Class entity.

 @see className
*/

/*#
   @method baseClass BOM
   @brief Gets the class of an item
   @return a Class entity.

   @see baseClass
   @see className
*/

class FALCON_DYN_CLASS BaseClass: public PseudoFunction
{
public:
   BaseClass();
   virtual ~BaseClass();
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

/* end of baseclass.h */

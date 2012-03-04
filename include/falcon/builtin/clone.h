/*
   FALCON - The Falcon Programming Language.
   FILE: clone.h

   Falcon core module -- clone function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 21:49:29 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_CLONE_H
#define	FALCON_CORE_CLONE_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function clone
*/

/*#
   @method clone BOM

*/

class FALCON_DYN_CLASS Clone: public PseudoFunction
{
public:
   Clone();
   virtual ~Clone();
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

#endif	/* FALCON_CORE_LEN_H */

/* end of clone.h */

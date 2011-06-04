/*
   FALCON - The Falcon Programming Language.
   FILE: typeid.h

   Implementation of typeId function or method.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 23:49:50 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_TYPEID_H_
#define _FALCON_TYPEID_H_

#include <falcon/pseudofunc.h>

namespace Falcon {
namespace Ext {

class FALCON_DYN_CLASS TypeId: public PseudoFunction
{
public:
   TypeId();
   virtual ~TypeId();
   virtual void apply( VMachine* vm, int32 nParams );

private:

   class Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      static void apply_( const PStep* ps, VMachine* vm );

   };

   Invoke m_invoke;
};

}
}

#endif

/* end of typeid.h */

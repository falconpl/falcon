/*
   FALCON - The Falcon Programming Language.
   FILE: len.h

   Falcon core module -- len function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 04 Jun 2011 20:52:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_LEN_H
#define	FALCON_CORE_LEN_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function len
   @brief Retrieves the length of a collection
   @param item an item of any kind
   @return the count of items in the sequence, or 0.

   The returned value represent the "size" of the item passed as a parameter.
   The number is consistent with the object type: in case of a string, it
   represents the count of characters, in case of arrays or dictionaries it
   represents the number of elements, in all the other cases the returned
   value is 0.
*/

/*#
   @method len BOM

   @brief Retrieves the length of a collection
   @return the count of items in the sequence, or 0.

   The returned value represent the "size" of this item.
   @see len
*/

class FALCON_DYN_CLASS Len: public PseudoFunction
{
public:
   Len();
   virtual ~Len();
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

/* end of len.h */

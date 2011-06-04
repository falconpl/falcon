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

#ifndef FALCON_CORE_COMPARE_H
#define	FALCON_CORE_COMPARE_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @method compare BOM
   @brief Performs a lexicographical comparison.
   @param item The item to which this object must be compared.
   @return -1, 0 or 1 depending on the comparation result.

   Performs a lexicographical comparison between the self item and the
   item passed as a parameter. If the item is found smaller than the parameter,
   it returns -1; if the item is greater than the parameter, it returns 1.
   If the two items are equal, it returns 0.

   The compare method, if overloaded, is used by the Virtual Machine to perform
   tests on unknown types (i.e. objects), and to sort dictionary keys.

   Item different by type are ordered by their type ID, as indicated in the
   documentation of the @a typeOf core function.

   By default, string comparison is performed in UNICODE character order,
   and objects, classes, vectors, and dictionaries are ordered by their
   internal pointer address.
*/

/*#
   @function compare
   @brief Performs a lexicographical comparison.
   @param item The item to which this object must be compared.
   @param item2 The item to which this object must be compared.
   @return -1, 0 or 1 depending on the comparation result.

   Performs a lexicographical comparison between the self item and the
   item passed as a parameter. If the item is found smaller than the parameter,
   it returns -1; if the item is greater than the parameter, it returns 1.
   If the two items are equal, it returns 0.

   The compare method, if overloaded, is used by the Virtual Machine to perform
   tests on unknown types (i.e. objects), and to sort dictionary keys.

   Item different by type are ordered by their type ID, as indicated in the
   documentation of the @a typeOf core function.

   By default, string comparison is performed in UNICODE character order,
   and objects, classes, vectors, and dictionaries are ordered by their
   internal pointer address.
*/

class FALCON_DYN_CLASS Compare: public PseudoFunction
{
public:
   Compare();
   virtual ~Compare();
   virtual void apply( VMachine* vm, int32 nParams );

private:
   class NextStep: public PStep
   {
   public:
      NextStep() { apply = apply_; }
      static void apply_( const PStep* ps, VMachine* vm );
   };

   class Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      static void apply_( const PStep* ps, VMachine* vm );

   };

   NextStep m_next;
   Invoke m_invoke;
};

}
}

#endif	/* FALCON_CORE_COMPARE_H */

/* end of len.h */

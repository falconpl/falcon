/*
   FALCON - The Falcon Programming Language.
   FILE: pstep.h

   Common interface to VM processing step.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 03 Jan 2011 18:01:18 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_PSTEP_H
#define FALCON_PSTEP_H

#include <falcon/setup.h>
#include <falcon/basealloc.h>
#include <falcon/string.h>

namespace Falcon {

/** Common interface to VM processing step.
 *
 * This is the common base class for statements, expressions and statement sequences,
 * which are the basic components of the Virtual Machine code processing.
 *
 * These are the 3 basic elements that can be processed by the virtual machine.
 * The VM scans its code stack, calling the apply() method of the topmost element.
 * When a sequence "apply" is invoked, it starts calling the perform() method of all its
 * members; they can be performed on the spot, acting immediately on the virtual machine,
 * or push themselves in the code stack and return the control to the calling VM.
 *

 *
 * Expressions can be also processed by statements that can call their eval() member directly.
 */
class FALCON_DYN_CLASS PStep: public BaseAlloc
{
public:
   inline PStep() {};

   inline virtual ~PStep() {}

   /* Called by the virtual machine when it encounters a statement or expression in a sequence. */
   virtual void perform( VMachine* vm ) const = 0;

   /* Called by the virtual machine when it encounters a step on its code stack. */
   virtual void apply( VMachine* vm ) const = 0;

   /** Convert into a string */
   inline const String toString() const
   {
      String temp;
      toString( temp );
      return temp;
   }

   /** Convert into a string.
    *
    * The default base class function does nothing. This is useful for
    * pstep that are not part of the syntactic tree, but just of the
    * VM code.
    * */
   inline virtual void toString( String& ) const {};
};

}

#endif

/* end of pstep.h */

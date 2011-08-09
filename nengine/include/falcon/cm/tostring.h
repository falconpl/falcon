/*
   FALCON - The Falcon Programming Language.
   FILE: tostring.h

   Falcon core module -- tostring function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:20:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_TOSTRING_H
#define	FALCON_CORE_TOSTRING_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function toString
   @brief Returns a string representation of the item.
   @param item The item to be converted to string.
   @optparam format Specific object format.
   @return the string representation of the item.

   This function is useful to convert an unknown value in a string. The item may be any kind of Falcon
   item; the following rules apply:
      - Nil items are represented as "<NIL>"
      - Integers are converted in base 10.
      - Floating point values are converted in base 10 with a default precision of 6;
        numprec may be specified to change the default precision.
      - Array and dictionaries are represented as "Array of 'n' elements" or "Dictionary of 'n' elements".
      - Strings are copied.
      - Objects are represented as "Object 'name of class'", but if a toString() method is provided by the object,
        that one is called instead.
      - Classes and other kind of opaque items are rendered with their names.

   This function is not meant to provide complex applications with pretty-print facilities, but just to provide
   simple scripts with a simple and consistent output facility.

   If a @b format parameter is given, the format will be passed unparsed to toString() methods of underlying
   items.

   @see Format
*/

/*#
   @method toString BOM
   @brief Coverts the object to string.
   @optparam format Optional object-specific format string.

   Calling this BOM method is equivalent to call toString() core function
   passing the item as the first parameter.

   Returns a string representation of the given item. If applied on strings,
   it returns the string as is, while it converts numbers with default
   internal conversion. Ranges are represented as "[N:M:S]" where N and M are respectively
   lower and higher limits of the range, and S is the step. Nil values are represented as
   "Nil".

   The format parameter is not a Falcon format specification, but a specific optional
   object-specific format that may be passed to objects willing to use them.
   In example, the TimeStamp class uses this parameter to format its string
   representation.
*/

class FALCON_DYN_CLASS ToString: public Function
{
public:
   ToString();
   virtual ~ToString();
   virtual void invoke( VMContext* ctx, int32 nParams );

private:

   class FALCON_DYN_CLASS Next: public PStep
   {
   public:
      Next() { apply = apply_; }
      virtual ~Next() {}
      static void apply_( const PStep* ps, VMContext* ctx );

   };

   Next m_next;
};

}
}

#endif	/* FALCON_CORE_TOSTRING_H */

/* end of tostring.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: describe.h

   Falcon core module -- len function/method
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jun 2011 20:23:04 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_DESCRIBE_H
#define	FALCON_CORE_DESCRIBE_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {


/*#
   @function describe
   @param item The item to be inspected.
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @brief Returns the deep contents of an item on a string representation.

   This function returns a string containing a representation of the given item.
   If the item is deep (an array, an instance, a dictionary) the contents are
   also passed through this function.

   This function traverses arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a @b maxLen parameter.

   You may create personalized inspect functions using forward bindings, like
   the following:
   @code
   compactDescribe = .[describe depth|1 maxLen|15]
   @endcode
*/

/*#
   @method describe BOM
   @optparam depth Maximum inspect depth.
   @optparam maxLen Limit the display size of possibly very long items as i.e. strings or membufs.
   @brief Returns the deep contents of an item on a string representation.

   This method returns a string containing a representation of this item.
   If the item is deep (an array, an instance, a dictionary) the contents are
   also passed through this function.

   This method traverses arrays and items deeply; there isn't any protection
   against circular references, which may cause endless loop. However, the
   default maximum depth is 3, which is a good depth for debugging (goes deep,
   but doesn't dig beyond average interesting points). Set to -1 to have
   infinite depth.

   By default, only the first 60 characters of strings and elements of membufs
   are displayed. You may change this default by providing a @b maxLen parameter.
*/


class FALCON_DYN_CLASS Describe: public Function
{
public:
   Describe();
   virtual ~Describe();
   virtual void invoke( VMContext* ctx, int32 nParams );
};

}
}

#endif	/* FALCON_CORE_DESCRIBE_H */

/* end of describe.h */

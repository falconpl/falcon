/*
   FALCON - The Falcon Programming Language.
   FILE: pagedict_ext.cpp

   Page dictionary extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:17:31 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/pagedict.h>

/*#

*/
namespace Falcon {
namespace core {

/*#
   @function PageDict
   @ingroup general_purpose
   @brief Creates a paged dictionary (which is internally represented as a B-Tree).
   @param pageSize size of pages expressed in maximum items.
   @return A new dictionary.

   The function returns a Falcon dictionary that can be handled exactly as a normal
   dictionary. The difference is only in the internal management of memory allocation
   and tree balance. Default Falcon dictionaries (the ones created with the "[=>]"
   operator) are internally represented as paired linear vectors of ordered entries.
   They are extremely efficient to store a relatively small set of data, whose size,
   and possibly key order, is known in advance. As this is exactly the condition under
   which source level dictionary are created, this way to store dictionary is the
   default in Falcon. The drawback is that if the data grows beyond a critical mass
   linear dictionary may become sluggishly slow and hang down the whole VM processing.

   This function, which is actually a class factory function (this is the reason why
   its name begins in uppercase), returns an empty Falcon dictionary that is internally
   organized as a B-Tree structure. At a marginal cost in term of memory with respect
   to the mere storage of falcon items, which is used as spare and growth area, this
   structure offer high performances on medium to large amount of data to be ordered
   and searched. Empirical tests in Falcon language showed that this structure can
   scale up easily to several millions items.

   In general, if a Falcon dictionary is meant to store large data, above five to ten
   thousands elements, or if the size of stored data is not known in advance, using
   this structure instead of the default Falcon dictionaries is highly advisable.
*/
FALCON_FUNC  PageDict( ::Falcon::VMachine *vm )
{
   Item *i_pageSize = vm->param(0);

   if( i_pageSize != 0 && ! i_pageSize->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params ).origin( e_orig_runtime ).extra( "[N]" ) );
   }

   uint32 pageSize = (uint32)( i_pageSize == 0 ? 33 : (uint32)i_pageSize->forceInteger() );
   CoreDict *cd = new CoreDict( new ::Falcon::PageDict( pageSize ) );
   vm->retval( cd );
}

}
}

/* end of pagedict_ext.cpp */

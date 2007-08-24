/*
   FALCON - The Falcon Programming Language.
   FILE: $FILE$.h
   $Id: uintpair.h,v 1.1.1.1 2006/10/08 15:05:39 gian Exp $
   
   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio ago 26 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)
   
   See LICENSE file for licensing details. 
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#ifndef flc_UINTPAIR_H
#define flc_UINTPAIR_H

#include <falcon/types.h>
namespace Falcon {

/** Prefect hash base structure.
   Perfect hashes are hashes that has not duplicate hash values by
   definition. They are used in switch() statements and class properties.
*/
template< class _T >
class uint_pair
{
public:
   uint32 first;
   _T second;
   
   uint_pair( uint32 f, _T s ):
      first(f),
      second(s)
   {}
   
};

}

#endif

/* end of flc_uintpair.h */

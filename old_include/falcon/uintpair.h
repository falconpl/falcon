/*
   FALCON - The Falcon Programming Language.
   FILE: $FILE$.h
   
   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: gio ago 26 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)
   
   See LICENSE file for licensing details. 
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

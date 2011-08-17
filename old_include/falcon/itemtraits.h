/*
   FALCON - The Falcon Programming Language.
   FILE: itemtraits.h

   Traits for putting items in generic containers
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ott 29 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Traits for putting items in generic containers
*/

#ifndef flc_itemtraits_H
#define flc_itemtraits_H

#include <falcon/setup.h>
#include <falcon/item.h>
#include <falcon/traits.h>

namespace Falcon {

class VMachine;

class FALCON_DYN_CLASS ItemTraits: public ElementTraits
{
public:

   virtual uint32 memSize() const;
   virtual void init( void *itemZone ) const;
   virtual void copy( void *targetZone, const void *sourceZone ) const;
   virtual int compare( const void *first, const void *second ) const;
   virtual void destroy( void *item ) const;
   virtual bool owning() const;
};



namespace traits {
   extern ItemTraits &t_item();
}


}

#endif

/* end of itemtraits.h */

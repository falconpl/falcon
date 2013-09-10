/*
   FALCON - The Falcon Programming Language.
   FILE: diskmpxfactory.h

   Traits for plain local disk files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 09:38:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DISK_MPXFACT_TRAITS_
#define _FALCON_DISK_MPXFACT_TRAITS_

#include <falcon/multiplex.h>

namespace Falcon {

class FALCON_DYN_CLASS DiskMpxFactory: public Multiplex::Factory
{
public:
   DiskMpxFactory()
   {}
   virtual ~DiskMpxFactory() {}

   virtual Multiplex* create( Selector* master ) const;

private:

   class MPX;
};

}

#endif

/* end of diskmpxfactory.h */

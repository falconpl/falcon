/*
   FALCON - The Falcon Programming Language.
   FILE: diskfiletraits.h

   Traits for plain local disk files.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 02 Mar 2013 09:38:34 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_DISK_FILE_TRAITS_
#define _FALCON_DISK_FILE_TRAITS_

#include <falcon/streamtraits.h>

namespace Falcon {

class FALCON_DYN_CLASS DiskFileTraits: public StreamTraits
{
public:
   /**
    * \param name A symbolic name for this kind of streams.
    * \param module A module where this multiplex code is stored.
    *
    * \note  The module is used for back-reference and keep alive marks.
    * Usually, the value is implicit, in the sense that the code is in a module
    * it knows, so it's not necessary to get this
    */
   DiskFileTraits():
      StreamTraits("DiskFileTraits", 0)
   {}

   virtual ~DiskFileTraits() {}

   virtual Multiplex* multiplex( Selector* master ) const;

private:

   class MPX;
};

}

#endif

/* end of diskfiletraits.h */

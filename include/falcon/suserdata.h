/*
   FALCON - The Falcon Programming Language.
   FILE: suserdata.h

   Embeddable falcon object user data - shared version
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 20 Mar 2008 21:20:52 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Embeddable falcon object user data.
*/

#ifndef flc_suserdata_H
#define flc_suserdata_H

#include <falcon/userdata.h>
#include <falcon/garbageable.h>

namespace Falcon {

/** Embeddable falcon object user data - shared version.
   This class is known by CoreObjects to be shared among other
   VM instances.

   They don't dispose of it when they are collected; instead,
   instances of this class are disposed by the garbage collector
   itself.

   This class is suitable to be used when more than one CoreObject
   may reference it.
*/

class FALCON_DYN_CLASS SharedUserData: public UserData, public Garbageable
{

public:
   SharedUserData( VMachine *vm );
   virtual bool shared() const;
   virtual void gcMark( MemPool *mp );
};

}

#endif

/* end of suserdata.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: multiclass_private.h

   Base class for classes holding more subclasses -- engine private part
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 19 Jul 2011 06:35:06 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/
#ifndef MULTICLASS_PRIVATE_H
#define MULTICLASS_PRIVATE_H

#include <falcon/classes/classmulti.h>

#include <string>
#include <map>

namespace Falcon {

class ClassMulti::Private_base
{
public:
   typedef std::map<String, Property> PropMap;
   PropMap m_props;

   Private_base() {}
   virtual ~Private_base() {}
};

}

#endif	/* FALCON_MULTICLASS_PRIVATE_H */

/* end of multiclass_private.h */

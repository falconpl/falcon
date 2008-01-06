/*
 * FALCON - The Falcon Programming Language.
 * FILE: bommap.h
 *
 * Falcon bill of materials map
 * -------------------------------------------------------------------
 * Author: Jeremy Cowgar
 * Begin: Sat Jan  5 2008
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2008 the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 * In order to use this file in its compiled form, this source or
 * part of it you have to read, understand and accept the conditions
 * that are stated in the LICENSE file that comes boundled with this
 * package.
 */

#ifndef BOMMAP_H
#define BOMMAP_H

#include <falcon/genericmap.h>
#include <falcon/traits.h>

#include <string.h>

namespace Falcon {

   class BomMap : public Map {
   public:
      BomMap() : Map( &traits::t_string, &traits::t_int ) {}
      void add( const String &key ) {
         int s = (int) size();
         insert( &key, &s );
      }
   };

}

#endif /* BOMMAP_H */

/* end of file bommap.h */


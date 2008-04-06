/*
   FALCON - The Falcon Programming Language.
   FILE: flc_deptab.h

   Dependency table declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 8 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_DEPTAB_H
#define flc_DEPTAB_H

#include <falcon/string.h>
#include <falcon/genericlist.h>

/** \file
   Dependency table support for modules - header - .
*/

namespace Falcon {

class Module;
class Stream;

/** Module dependency table.
   Actually it's just a string list supporting module-aware serialization.
   The strings are actually held in the module string table.
*/
class FALCON_DYN_CLASS DependTable: public List
{
public:
   DependTable() {}

   bool save( Stream *out ) const ;
   bool load( Module *mod, Stream *in );

};

}

#endif

/* end of flc_deptab.h */

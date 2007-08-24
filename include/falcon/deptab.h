/*
   FALCON - The Falcon Programming Language.
   FILE: flc_deptab.h
   $Id: deptab.h,v 1.2 2006/10/15 20:21:50 gian Exp $

   Dependency table declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 8 2004
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
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

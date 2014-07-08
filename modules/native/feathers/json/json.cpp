/*
   FALCON - The Falcon Programming Language
   FILE: json.cpp

   JSON module main file
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 06 Sep 2008 09:48:38 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


/*#
   @module json JSON support
   @ingroup feathers
   @brief JavaScript Object Notation interface.

   This module exposes functions to dump and load variables
   encoded in JSON format. See @link http://json.org for more details.

   The module is meant to be included in a namespace (for example, json);
   it's not advisable to load it via load. Instead, use the import
   directive as in the following example:

   @code
      import from json
      > JSON.encode( ["a", 1, 1.2] )
   @endcode
*/

/*#
   @beginmodule feathers.json
*/

#include "json_fm.h"

FALCON_MODULE_DECL
{

   Falcon::Module *self = new Falcon::Feathers::ModuleJSON();
   return self;
}


/* end of funcext.cpp */


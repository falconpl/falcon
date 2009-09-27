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
   @module feather_json json
   @brief JavaScript Object Notation interface.

   This module exposes function to dump and load variables
   encoded in JSON format. See @link http://json.org for more details.

   The module is meant to be included in a namespace (for example, json);
   it's not advisable to load it via load. Instead, use the import
   directive as in the following example:

   @code
      import from json
      > json.encode( ["a", 1, 1.2] )
   @endcode


   @beginmodule feather_json
*/

#include <falcon/module.h>
#include <falcon/srv/json_srv.h>
#include "json_ext.h"
#include "json_st.h"

#include "version.h"

static Falcon::JSONService s_theJSONService;

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   Falcon::Module *self = new Falcon::Module();
   self->name( "json" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "json_st.h"

   self->addExtFunc( "encode", &Falcon::Ext::json_encode )->
      addParam("item")->addParam("stream");

   self->addExtFunc( "decode", &Falcon::Ext::json_decode )->
      addParam("source");

   self->addExtFunc( "apply", &Falcon::Ext::json_apply )->
      addParam("source")->addParam("item");

   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *jerr_cls = self->addClass( "JSONError", &Falcon::Ext::JSONError_init );
   jerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );

   //======================================
   // Subscribe the service
   //
   self->publishService( &s_theJSONService );

   return self;
}


/* end of funcext.cpp */


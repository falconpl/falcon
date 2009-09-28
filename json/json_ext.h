/*
   FALCON - The Falcon Programming Language.
   FILE: json_ext.h

   JSON transport format interface -
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 27 Sep 2009 18:28:44 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler module main file - extension definitions.
*/

#ifndef flc_json_ext_H
#define flc_json_ext_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error_base.h>
#include <falcon/error.h>

#ifndef FALCON_JSON_ERROR_BASE
   #define FALCON_JSON_ERROR_BASE        1210
#endif

#define FALCON_JSON_NOT_CODEABLE    (FALCON_JSON_ERROR_BASE + 0)
#define FALCON_JSON_NOT_DECODABLE   (FALCON_JSON_ERROR_BASE + 1)
#define FALCON_JSON_NOT_APPLY       (FALCON_JSON_ERROR_BASE + 2)

namespace Falcon {
namespace Ext {

FALCON_FUNC  json_encode ( ::Falcon::VMachine *vm );
FALCON_FUNC  json_decode ( ::Falcon::VMachine *vm );
FALCON_FUNC  json_apply ( ::Falcon::VMachine *vm );

class JSONError: public ::Falcon::Error
{
public:
   JSONError():
      Error( "JSONError" )
   {}

   JSONError( const ErrorParam &params  ):
      Error( "JSONError", params )
      {}
};

FALCON_FUNC  JSONError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of json_ext.h */

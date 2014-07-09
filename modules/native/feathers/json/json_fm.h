/*
   FALCON - The Falcon Programming Language.
   FILE: json_fm.h

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

#ifndef FALCON_FEATHERS_JSON_FM_H
#define FALCON_FEATHERS_JSON_FM_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error_base.h>
#include <falcon/error.h>

#define FALCON_FEATHER_JSON_NAME "json"

#ifndef FALCON_JSON_ERROR_BASE
   #define FALCON_JSON_ERROR_BASE        1210
#endif

#define FALCON_JSON_NOT_CODEABLE    (FALCON_JSON_ERROR_BASE + 0)
#define FALCON_JSON_NOT_CODEABLE_DESC "Given object cannot be rendered as json string"

#define FALCON_JSON_NOT_DECODABLE   (FALCON_JSON_ERROR_BASE + 1)
#define FALCON_JSON_NOT_DECODABLE_DESC "JSON Data not applicable to given object."

#define FALCON_JSON_NOT_APPLY       (FALCON_JSON_ERROR_BASE + 2)
#define FALCON_JSON_NOT_APPLY_DESC "Data is not in json format"

namespace Falcon {
namespace Feathers {

/*#
   @class JSONError
   @brief Error generated after error conditions on JSON operations.
   @optparam code The error code
   @optparam desc The description for the error code
   @optparam extra Extra information specifying the error conditions.
   @from Error( code, desc, extra )
*/

FALCON_DECLARE_ERROR( JSONError )


class ClassJSON: public Class
{
public:
   ClassJSON();
   virtual ~ClassJSON();

   virtual int64 occupiedMemory( void* instance ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;

   // we don't have init: just static methods
};

class ModuleJSON: public Module
{
public:
   ModuleJSON();
   virtual ~ModuleJSON();
};

}
}

#endif

/* end of json_fm.h */

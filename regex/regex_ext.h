/*
   FALCON - The Falcon Programming Language.
   FILE: regex_ext.h

   Short description
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   Process module -- Falcon interface functions
   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Regular expression module -- Falcon interface functions
   This is the module declaration file.
*/

#ifndef flc_regex_ext_H
#define flc_regex_ext_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>

namespace Falcon {
namespace Ext {

FALCON_FUNC Regex_init( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_study( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_match( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_grab( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_find( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_findAll( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_findAllOverlapped( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_replace( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_replaceAll( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_capturedCount( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_captured( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_compare( ::Falcon::VMachine *vm );
FALCON_FUNC Regex_version( ::Falcon::VMachine *vm );

class RegexError: public ::Falcon::Error
{
public:
   RegexError():
      Error( "RegexError" )
   {}

   RegexError( const ErrorParam &params  ):
      Error( "RegexError", params )
      {}
};

FALCON_FUNC  RegexError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of regex_ext.h */

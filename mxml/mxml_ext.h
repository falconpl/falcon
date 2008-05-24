/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_ext.h

   Minimal XML module main file - extension definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Mar 2008 18:30:01 +0100

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Minimal XML module main file - extension definitions.
*/

#ifndef flc_mxml_ext_H
#define flc_mxml_ext_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_MXML_ERROR_BASE
   #define FALCON_MXML_ERROR_BASE            1120
#endif

namespace Falcon {
namespace Ext {

FALCON_FUNC MXMLDocument_init( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_deserialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_style( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_top( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_root( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_find( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_findNext( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_findPath( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_findPathNext( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_save( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_load( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_setEncoding( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_getEncoding( ::Falcon::VMachine *vm );

FALCON_FUNC MXMLNode_init( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_deserialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_nodeType( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_name( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_data( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_setAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_getAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_getAttribs( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_getChildren( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_unlink( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_removeChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_parent( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_firstChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_nextSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_lastChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_addBelow( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertBelow( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertBefore( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertAfter( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_depth( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_path( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_clone( ::Falcon::VMachine *vm );


class MXMLError: public ::Falcon::Error
{
public:
   MXMLError():
      Error( "MXMLError" )
   {}

   MXMLError( const ErrorParam &params  ):
      Error( "MXMLError", params )
      {}
};

FALCON_FUNC  MXMLError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of mxml_ext.h */

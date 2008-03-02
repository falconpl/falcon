/*
   FALCON - The Falcon Programming Language.
   FILE: mxml_ext.h

   Compiler module main file - extension definitions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 02 Mar 2008 18:30:01 +0100
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Compiler module main file - extension definitions.
*/

#ifndef flc_compiler_ext_H
#define flc_compiler_ext_H

#include <falcon/setup.h>
#include <falcon/module.h>

namespace Falcon {

namespace Ext {

FALCON_FUNC MXMLDocument_init( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_deserialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_style( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_root( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_find( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_findPath( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_save( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLDocument_load( ::Falcon::VMachine *vm );

FALCON_FUNC MXMLNode_init( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_deserialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_nodeType( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_name( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_data( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_setAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_getAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_hasAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_hasAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_hasAttribute( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_removeChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_parent( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_firstChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_nextSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_lastChild( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_prevSibling( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_addBelow( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertBelo( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertBefore( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_insertAfter( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_depth( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_path( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_clone( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_serialize( ::Falcon::VMachine *vm );
FALCON_FUNC MXMLNode_deserialize( ::Falcon::VMachine *vm );

}
}

#endif

/* end of compiler_ext.h */

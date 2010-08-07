/*
   FALCON - The Falcon Programming Language.
   FILE: socket_ext.cpp

   Falcon VM interface to confparser module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Falcon VM interface to confparser module -- header.
*/


#ifndef FLC_CONFPARSER_EXT_H
#define FLC_CONFPARSER_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>

#include <falcon/error_base.h>

#ifndef FALCON_CONFPARSER_ERROR_BASE
   #define FALCON_CONFPARSER_ERROR_BASE        1110
#endif

#define FALCP_ERR_INVFORMAT  (FALCON_CONFPARSER_ERROR_BASE + 0)
#define FALCP_ERR_STORE      (FALCON_CONFPARSER_ERROR_BASE + 1)

namespace Falcon {
namespace Ext {

// ==============================================
// Class ConfParser
// ==============================================
FALCON_FUNC  ConfParser_init( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_read( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_write( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_get( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getOne( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getMultiple( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getSections( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getKeys( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getCategoryKeys( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getCategory( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_removeCategory( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_getDictionary( ::Falcon::VMachine *vm );

FALCON_FUNC  ConfParser_add( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_set( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_remove( ::Falcon::VMachine *vm );

FALCON_FUNC  ConfParser_addSection( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_removeSection( ::Falcon::VMachine *vm );
FALCON_FUNC  ConfParser_clearMain( ::Falcon::VMachine *vm );

}
}

#endif

/* end of socket_ext.h */

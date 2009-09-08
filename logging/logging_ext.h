/*
   FALCON - The Falcon Programming Language.
   FILE: logging_ext.cpp

   Falcon VM interface to logging module -- header.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/



#ifndef FLC_LOGGING_EXT_H
#define FLC_LOGGING_EXT_H

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/coreobject.h>

#include <falcon/error.h>
#include <falcon/cclass.h>

#include <falcon/error_base.h>

namespace Falcon {

//=====================================================
// CoreLogArea
//=====================================================

namespace Ext {

// ==============================================
// Class LogArea
// ==============================================
FALCON_FUNC  LogArea_init( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_add( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_remove( ::Falcon::VMachine *vm );
FALCON_FUNC  LogArea_log( ::Falcon::VMachine *vm );

FALCON_FUNC  GeneralLog_init( ::Falcon::VMachine *vm );

FALCON_FUNC  LogChannel_init( ::Falcon::VMachine *vm );
FALCON_FUNC  LogChannel_level( ::Falcon::VMachine *vm );
FALCON_FUNC  LogChannel_format( ::Falcon::VMachine *vm );

FALCON_FUNC  LogChannelStream_init( ::Falcon::VMachine *vm );
FALCON_FUNC  LogChannelStream_flushAll( ::Falcon::VMachine *vm );

FALCON_FUNC  LogChannelSyslog_init( ::Falcon::VMachine *vm );

// ==============================================
// Generic area functions
// ==============================================
FALCON_FUNC  glog( ::Falcon::VMachine *vm );
FALCON_FUNC  glogf( ::Falcon::VMachine *vm );
FALCON_FUNC  gloge( ::Falcon::VMachine *vm );
FALCON_FUNC  glogw( ::Falcon::VMachine *vm );
FALCON_FUNC  glogi( ::Falcon::VMachine *vm );
FALCON_FUNC  glogd( ::Falcon::VMachine *vm );

}
}

#endif

/* end of logging_ext.h */

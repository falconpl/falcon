/*
   FALCON - The Falcon Programming Language.
   FILE: threading_ext.h

   Threading module binding extensions.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 10 Apr 2008 00:44:09 +0200

   -------------------------------------------------------------------
   (C) Copyright 2008: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Threading module binding extensions.
*/

#ifndef flc_threading_ext_H
#define flc_threading_ext_H

#include <falcon/setup.h>
#include <falcon/error_base.h>
#include "threading_ext.h"
#include "threading_mod.h"

#ifndef FALCON_THREADING_ERROR_BASE
   #define FALCON_THREADING_ERROR_BASE        2050
#endif

#define FALTH_ERR_NOTRUN      (FALCON_THREADING_ERROR_BASE + 0)
#define FALTH_ERR_RUNNING     (FALCON_THREADING_ERROR_BASE + 1)
#define FALTH_ERR_PREPARE     (FALCON_THREADING_ERROR_BASE + 2)
#define FALTH_ERR_START       (FALCON_THREADING_ERROR_BASE + 3)
#define FALTH_ERR_NOTRUNNING  (FALCON_THREADING_ERROR_BASE + 4)
#define FALTH_ERR_NOTTERM     (FALCON_THREADING_ERROR_BASE + 5)
#define FALTH_ERR_JOIN        (FALCON_THREADING_ERROR_BASE + 6)
#define FALTH_ERR_JOINE       (FALCON_THREADING_ERROR_BASE + 7)
#define FALTH_ERR_QEMPTY      (FALCON_THREADING_ERROR_BASE + 8)
#define FALTH_ERR_DESERIAL    (FALCON_THREADING_ERROR_BASE + 9)

namespace Falcon {
namespace Ext {

//=====================================================
// Threading nametion
//

FALCON_FUNC Threading_wait( VMachine *vm );
FALCON_FUNC Threading_vwait( VMachine *vm );
FALCON_FUNC Threading_getCurrentID( VMachine *vm );
FALCON_FUNC Threading_getCurrent( VMachine *vm );
FALCON_FUNC Threading_sameThread( VMachine *vm );
FALCON_FUNC Threading_start( VMachine *vm );

//=====================================================
// Thread class
//
FALCON_FUNC Thread_init( VMachine *vm );
FALCON_FUNC Thread_start( VMachine *vm );
FALCON_FUNC Thread_stop( VMachine *vm );
FALCON_FUNC Thread_detach( VMachine *vm );
FALCON_FUNC Thread_wait( VMachine *vm );
FALCON_FUNC Thread_vwait( VMachine *vm );
FALCON_FUNC Thread_getThreadID( VMachine *vm );
FALCON_FUNC Thread_sameThread( VMachine *vm );
FALCON_FUNC Thread_getSystemID( VMachine *vm );
FALCON_FUNC Thread_getName( VMachine *vm );
FALCON_FUNC Thread_setName( VMachine *vm );
FALCON_FUNC Thread_toString( VMachine *vm );

FALCON_FUNC Thread_getError( VMachine *vm );
FALCON_FUNC Thread_getReturn( VMachine *vm );
FALCON_FUNC Thread_hadError( VMachine *vm );
FALCON_FUNC Thread_terminated( VMachine *vm );
FALCON_FUNC Thread_detached( VMachine *vm );

FALCON_FUNC Thread_join( VMachine *vm );

//=====================================================
// Waitable
//
FALCON_FUNC Waitable_release( VMachine *vm );

//=====================================================
// Grant
//
FALCON_FUNC Grant_init( VMachine *vm );

//=====================================================
// Barrier
//
FALCON_FUNC Barrier_init( VMachine *vm );
FALCON_FUNC Barrier_open( VMachine *vm );
FALCON_FUNC Barrier_close( VMachine *vm );

//=====================================================
// Event
//
FALCON_FUNC Event_init( VMachine *vm );
FALCON_FUNC Event_set( VMachine *vm );
FALCON_FUNC Event_reset( VMachine *vm );

//=====================================================
// Counter
//
FALCON_FUNC SyncCounter_init( VMachine *vm );
FALCON_FUNC SyncCounter_post( VMachine *vm );

//=====================================================
// SyncQueue_init
//
FALCON_FUNC SyncQueue_init( VMachine *vm );
FALCON_FUNC SyncQueue_push( VMachine *vm );
FALCON_FUNC SyncQueue_pushFront( VMachine *vm );
FALCON_FUNC SyncQueue_pop( VMachine *vm );
FALCON_FUNC SyncQueue_popFront( VMachine *vm );
FALCON_FUNC SyncQueue_empty( VMachine *vm );
FALCON_FUNC SyncQueue_size( VMachine *vm );

//=====================================================
// ThreadError
//
FALCON_FUNC ThreadError_init( VMachine *vm );

//=====================================================
// JoinError
//
FALCON_FUNC JoinError_init( VMachine *vm );

}
}

#endif

/* end of threading_ext.h */

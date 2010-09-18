/*
   FALCON - The Falcon Programming Language.
   FILE: process_ext.h

   Process module -- Falcon interface functions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process module -- Falcon interface functions
   This is the module declaration file.
*/

#ifndef flc_process_ext_H
#define flc_process_ext_H

#include <falcon/module.h>
#include <falcon/error.h>
#include <falcon/error_base.h>

#ifndef FALCON_PROCESS_ERROR_BASE
   #define FALCON_PROCESS_ERROR_BASE        1140
#endif

#define FALPROC_ERR_READLIST  (FALCON_PROCESS_ERROR_BASE + 0)
#define FALPROC_ERR_CLOSELIST  (FALCON_PROCESS_ERROR_BASE + 1)
#define FALPROC_ERR_CREATLIST  (FALCON_PROCESS_ERROR_BASE + 2)
#define FALPROC_ERR_CREATPROC  (FALCON_PROCESS_ERROR_BASE + 3)
#define FALPROC_ERR_WAIT      (FALCON_PROCESS_ERROR_BASE + 4)
#define FALPROC_ERR_TERM      (FALCON_PROCESS_ERROR_BASE + 5)

namespace Falcon {
namespace Ext {

FALCON_FUNC  falcon_system ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_systemCall ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_pread ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_exec ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_processId ( ::Falcon::VMachine *vm );
FALCON_FUNC  falcon_processKill ( ::Falcon::VMachine *vm );

/**
   Process( command, [sinkin, sinkout, sinkerr, mergeerr, background] )
*/
FALCON_FUNC  Process_init ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_wait ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_terminate ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_value ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_getInput ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_getOutput ( ::Falcon::VMachine *vm );
FALCON_FUNC  Process_getAux ( ::Falcon::VMachine *vm );

FALCON_FUNC  ProcessEnum_init  ( ::Falcon::VMachine *vm );
FALCON_FUNC  ProcessEnum_next  ( ::Falcon::VMachine *vm );
FALCON_FUNC  ProcessEnum_close  ( ::Falcon::VMachine *vm );

class ProcessError: public ::Falcon::Error
{
public:
   ProcessError():
      Error( "ProcessError" )
   {}

   ProcessError( const ErrorParam &params  ):
      Error( "ProcessError", params )
      {}
};

FALCON_FUNC  ProcessError_init ( ::Falcon::VMachine *vm );

}
}

#endif

/* end of process_ext.h */

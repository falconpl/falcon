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

namespace Falcon { namespace Ext {

FALCON_FUNC  process_system ( VMachine *vm );
FALCON_FUNC  process_systemCall ( VMachine *vm );
FALCON_FUNC  process_pread ( VMachine *vm );
FALCON_FUNC  process_exec ( VMachine *vm );
FALCON_FUNC  process_processId ( VMachine *vm );
FALCON_FUNC  process_processKill ( VMachine *vm );

/**
   Process( command, [sinkin, sinkout, sinkerr, mergeerr, background] )
*/
struct Process
{
   static FALCON_FUNC  init ( VMachine *vm );
   static FALCON_FUNC  wait ( VMachine *vm );
   static FALCON_FUNC  terminate ( VMachine *vm );
   static FALCON_FUNC  value ( VMachine *vm );
   static FALCON_FUNC  getInput ( VMachine *vm );
   static FALCON_FUNC  getOutput ( VMachine *vm );
   static FALCON_FUNC  getAux ( VMachine *vm );
   static void registerExtensions( Module* );
   static CoreObject* factory(const CoreClass* cls, void* user_data, bool );
};

struct ProcessEnum
{
   static FALCON_FUNC  init  ( VMachine *vm );
   static FALCON_FUNC  next  ( VMachine *vm );
   static FALCON_FUNC  close  ( VMachine *vm );
   static void registerExtensions( Module* );
   static CoreObject* factory(const CoreClass* cls, void* user_data, bool );
};

struct ProcessError : Falcon::Error
{
   ProcessError():
      Error( "ProcessError" )
   { }

   ProcessError( const ErrorParam &params  ):
      Error( "ProcessError", params )
   { }

   static FALCON_FUNC  init ( VMachine *vm );
   static void registerExtensions( Module* );
};


}} // ns Falcon::Ext

#endif

/* end of process_ext.h */

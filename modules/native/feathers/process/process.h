/*
   FALCON - The Falcon Programming Language.
   FILE: process.h

   System dependent module specifications for process module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 01 Sep 2013 19:10:59 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_FEATHERS_PROCESS_H
#define FALCON_FEATHERS_PROCESS_H

#include <falcon/error_base.h>

#ifndef FALCON_PROCESS_ERROR_BASE
   #define FALCON_PROCESS_ERROR_BASE        1140
#endif


#define FALCON_PROCESS_ERROR_ALREADY_OPEN (FALCON_PROCESS_ERROR_BASE+1)
#define FALCON_PROCESS_ERROR_ALREADY_OPEN_MSG "Process already open"

#define FALCON_PROCESS_ERROR_OPEN_PIPE (FALCON_PROCESS_ERROR_BASE+2)
#define FALCON_PROCESS_ERROR_OPEN_PIPE_MSG "Cannot allocate pipes towards child process"

#define FALCON_PROCESS_ERROR_TERMINATE (FALCON_PROCESS_ERROR_BASE+3)
#define FALCON_PROCESS_ERROR_TERMINATE_MSG "Cannot terminate process"


#define FALCON_PROCESS_MSG_errlist   "Error while reading the process list"
#define FALCON_PROCESS_MSG_errlist2  "Error while closing the process list"
#define FALCON_PROCESS_MSG_errlist3  "Error while creating the process list"
#define FALCON_PROCESS_MSG_allstr    "All the elements in the first parameter must be strings"
#define FALCON_PROCESS_MSG_prccreate "Can't open the process"
#define FALCON_PROCESS_MSG_waitfail  "Wait failed"
#define FALCON_PROCESS_MSG_termfail  "Terminate failed"

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/mt.h>

namespace Falcon {

}

#endif

/* end of process.h */

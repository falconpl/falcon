/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys_unix.h

   Unix implementation of process handle
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Unix implementation of process handle
*/

#ifndef flc_process_sys_unix_H
#define flc_process_sys_unix_H

#include <sys/types.h>
#include "process.h"

namespace Falcon { namespace Sys {

class PosixProcess: public Process
{
public:
   PosixProcess();
   ~PosixProcess();


   /*
    * Interface Implementation
    */
   Falcon::Stream* inputStream();
   Falcon::Stream* outputStream();
   Falcon::Stream* errorStream();
   //
   bool close();
   bool wait( bool block );
   bool terminate( bool severe = false );

   pid_t pid() const { return m_pid; }
private:
   friend bool openProcess(Process* ph, String** argList,
                           bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );

   int m_file_des_in[2];
   int m_file_des_out[2];
   int m_file_des_err[2];

   pid_t m_pid;
};

}} // ns Falcon::Sys

#endif

/* end of process_sys_unix.h */

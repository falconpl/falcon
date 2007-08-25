/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys_unix.h
   $Id: process_sys_unix.h,v 1.1.1.1 2006/10/08 15:05:09 gian Exp $

   Unix implementation of process handle
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005
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
   Unix implementation of process handle
*/

#ifndef flc_process_sys_unix_H
#define flc_process_sys_unix_H

#include <sys/types.h>
#include "process_sys.h"

namespace Falcon {

class FileService;

namespace Sys {

class UnixProcessHandle: public ProcessHandle
{
   int m_file_des_in[2];
   int m_file_des_out[2];
   int m_file_des_err[2];

   pid_t m_pid;

   friend ProcessHandle *openProcess( String **args, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );
public:
   UnixProcessHandle():
      ProcessHandle()
   {}

   virtual ~UnixProcessHandle();

   pid_t pid() const { return m_pid; }

   virtual ::Falcon::Stream *getInputStream();
   virtual ::Falcon::Stream *getOutputStream();
   virtual ::Falcon::Stream *getErrorStream();

   virtual bool close();
   virtual bool wait( bool block );
   virtual bool terminate( bool severe = false );
};

}
}

#endif

/* end of process_sys_unix.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys.h
   $Id: process_sys.h,v 1.3 2007/08/02 23:41:40 jonnymind Exp $

   System dependant module specifications for process module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006
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
   System dependant module specifications for process module
*/

#ifndef flc_process_sys_H
#define flc_process_sys_H

#include <falcon/userdata.h>

namespace Falcon {

class Stream;
class FileService;
class String;


namespace Sys {

class ProcessHandle: public UserData
{
   int m_procVal;
   int m_lastError;
   bool m_done;

public:
   ProcessHandle():
      m_done(false),
      m_lastError( 0 ),
      m_procVal( 0 )
   {}

   virtual ::Falcon::Stream *getInputStream() =0;
   virtual ::Falcon::Stream *getOutputStream() =0;
   virtual ::Falcon::Stream *getErrorStream() =0;

   virtual bool close() = 0;
   virtual bool wait( bool block ) = 0 ;
   virtual bool terminate( bool severe = false ) = 0;

   int processValue() const { return m_procVal; }
   void processValue( int val ) { m_procVal = val; }
   bool done() const { return m_done; }
   void done( bool d ) { m_done = d; }
   int lastError() const { return m_lastError; }
   void lastError( int val ) { m_lastError = val; }
};

class ProcessEnum: public UserData
{
   void *m_sysdata;

public:
   ProcessEnum();
   virtual ~ProcessEnum();

   /** Get next entry in the enum.
      \return -1 on error, 0 on done, 1 on next available.
   */
   int next( String &name, uint64 &pid, uint64 &ppid, String &path );
   bool close();
};

bool spawn( String **args, bool overlay, bool background, int *result );
const char *shellParam();
const char *shellName();
ProcessHandle *openProcess( String **args, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );

uint64 processId();
bool processKill( uint64 id );
bool processTerminate( uint64 id );

}
}

#endif

/* end of process_sys.h */

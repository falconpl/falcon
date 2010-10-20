/*
   FALCON - The Falcon Programming Language.
   FILE: process_sys.h

   System dependent module specifications for process module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   System dependent module specifications for process module
*/

#ifndef flc_process_sys_H
#define flc_process_sys_H

#include <falcon/stream.h>
#include <falcon/string.h>

namespace Falcon { namespace Sys {



class Process
{
   bool m_done;
   int m_lastError;
   int m_procVal;

public:
   Process():
       m_done(false),
       m_lastError( 0 ),
       m_procVal( 0 )
   {}
   virtual ~Process() { }

   /*
    * Interface
    */
   virtual Falcon::Stream* inputStream() = 0;
   virtual Falcon::Stream* outputStream() = 0;
   virtual Falcon::Stream* errorStream() = 0;
   //
   virtual bool close() = 0;
   virtual bool wait( bool block ) = 0 ;
   virtual bool terminate( bool severe = false ) = 0;


   int processValue() const { return m_procVal; }
   void processValue( int val ) { m_procVal = val; }
   bool done() const { return m_done; }
   void done( bool d ) { m_done = d; }
   int lastError() const { return m_lastError; }
   void lastError( int val ) { m_lastError = val; }

   static Process* factory();
};



class ProcessEnum
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
bool spawn_read( String **args, bool overlay, bool background, int *result, String *sOut );

const char *shellParam();
const char *shellName();
bool openProcess(Process* ph, String **args, bool sinkin, bool sinkout, bool sinkerr, bool mergeErr, bool bg );

uint64 processId();
bool processKill( uint64 id );
bool processTerminate( uint64 id );


}} // ns Falcon::Sys

#endif

/* end of process_sys.h */

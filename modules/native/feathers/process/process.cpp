/*
   FALCON - The Falcon Programming Language.
   FILE: process.cpp

   Process oriented extensions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat Jan 29 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process oriented extensions
*/

#include <falcon/module.h>
#include "mod/process.h"
#include "ext/process.h"
#include "process_st.h"

#include "version.h"

/*#
   @module feathers.process External process interface
   @brief Process enumeration and subprocess control.

   The @b process module provides several functions to manage processes in the system
   and to manage interprocess communication. Child processes can be created and
   managed via the Process class, which provides a set of streams that can be read
   and written synchronously and polled for activity.

   To provide this functionality, the Process class requires the service called
   "Stream" normally provided by the core module. Embedders that wish to allow the
   scripts to load this module should provide core to scripts, or provide otherwise
   the "Stream" service in the Virtual Machines that will run those scripts.
   Failing to do so will cause the stream oriented methods of the Process class to
   raise an error. The generic process management functions in the module, and also
   the generic child-process management methods in the Process class do not require
   this service, so embedders not willing to provide core may just turn overload the
   sensible Process methods or mask the Process class right away (or just ignore
   this fact letting the error to be raised in case of mis-usage).

   As the Falcon command line automatically loads the core module, this remark does not
   apply to stand-alone scripts.

   The Process module declares a ProcessError class, derived from core Error class,
   which doesn't change any of its property or method. The ProcessError class is
   used to mark errors raised by this module.

   While the OS typically ensures that none of the functions declared in this
   module may actually fail, unless system conditions are critical, every function
   or method in this module may raise this error in case of unexpected system
   behavior.

   @note To make the entities delcared in this module available to falcon scripts
      use the command:
      @code
         load process
      @endcode


   @beginmodule feathers.process
*/

FALCON_MODULE_DECL
{
   #define FALCON_DECLARE_MODULE self

   // initialize the module
   Falcon::Module *self = new Falcon::Module();
   self->name( "process" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "process_st.h"

   //============================================================
   // Minimal system api
   //
   self->addExtFunc( "system", Falcon::Ext::process_system )->
      addParam("command")->addParam("background");
   self->addExtFunc( "systemCall", Falcon::Ext::process_systemCall )->
      addParam("command")->addParam("background");
   self->addExtFunc( "pread", Falcon::Ext::process_pread )->
      addParam("command")->addParam("background");
   self->addExtFunc( "exec", Falcon::Ext::process_exec )->
      addParam("command");
   self->addExtFunc( "processId", Falcon::Ext::process_processId );
   self->addExtFunc( "processKill", Falcon::Ext::process_processKill )->
      addParam("pid")->addParam("severe");


   Falcon::Ext::ProcessEnum::registerExtensions(self);
   Falcon::Ext::Process::registerExtensions(self);
   Falcon::Ext::ProcessError::registerExtensions(self);



   //============================================================
   // Add the process attribute constants
   self->addConstant( "PROCESS_SINK_INPUT", (Falcon::int64) 0x1 );
   self->addConstant( "PROCESS_SINK_OUTPUT", (Falcon::int64) 0x2 );
   self->addConstant( "PROCESS_SINK_AUX", (Falcon::int64) 0x4 );
   self->addConstant( "PROCESS_MERGE_AUX", (Falcon::int64) 0x8 );
   self->addConstant( "PROCESS_BG", (Falcon::int64) 0x10 );
   self->addConstant( "PROCESS_USE_SHELL", (Falcon::int64) 0x20 );

   return self;
}


/* end of process.cpp */


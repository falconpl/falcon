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

#include "process_fm.h"


/*#
   @module process External process interface
   @ingroup process
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

   @note To make the entities declared in this module available to falcon scripts
      use the command:
      @code
         load process
      @endcode


   @beginmodule feathers.process
*/

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   // initialize the module
   Falcon::Module *self = new Falcon::Feathers::ModuleProcess();

   return self;
}

#endif

/* end of process.cpp */

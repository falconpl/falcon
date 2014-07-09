/*
   FALCON - The Falcon Programming Language.
   FILE: logging.cpp

   The logging module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 05 Sep 2009 17:21:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The confparser module - main file.
*/
#include "logging_fm.h"


/*#
   @module logging Logging facility
   @ingroup feathers
   @brief Multithread enabled logging facility.

   The @b logging module offers a very advanced facility for logging application
   messages to various device.

   It is based on a Area/Channel architecture.

   The @a LogArea class provides a logical
   subdivision of the semantic meaning of a log entry; sending a log to a certain
   area means that the entry has as certain meaning and function in the application
   architecture. For example, an application can have "security", "trace" and
   "status" areas; security area is meant to log informations and errors concerning
   security policy operations (login, logout, password change and so on). The status
   area may be dedicated to log internal operation status, as, for example failures
   in opening a database, performed queries or amount of records changed. The trace
   area may be specifically dedicated to debug or deep audit.

   Channels are physical or logical device managers, receiving log requests and
   delivering them on final media, rendered as their internal rules dictate.

   The user application posts a log entry to an area, and the area dispatches
   the entry to all the channels that are registered with that area. Channels can be
   registered with multiple areas, and they can be dynamically added to new
   areas or removed from areas they are currently connected to.

   Each message it's associated with a "level", also called severity, which
   determines how important the log entry is. Channels can be configured to
   perform log operations only to messages having at least a minimum level,
   that is, severe at least as a certain level. Lower level entries are ignored
   by the channel.

   The current logger module provides one default area, called @a GeneralLog, and
   three log channel classes:
   - @a LogChannelStream - writes entries on a Falcon stream separately opened.
   - @a LogChannelSyslog - sends entries to the system log facility. On POSIX
     systems, logs are sent to the Syslog daemon, while on MS-Windows systems
     they are sent to the Event Logger.
   - @a LogChannelFiles - Writes entries to a local file, and swaps or rotates
     the log file on need.

   @section feather_logging_performance Performance considerations

   The logger module is thought to perform the final part of the log operations
   in the most efficient way possible. Each channel performs the final log
   rendering and the dispatching on the underlying media on a separate thread,
   so the operation of generating a log for a certain channel is virtually
   non-blocking and relatively fast.

   Log operations involving fixed parameters are nearly no-ops, as in the following
   example:

   @code
   load logging
   log( 2000, "A very low priority level, unlikely to be ever logged" )
   @endcode

   Although there is a cost in calling the @a glog function (logging on the GenericLog
   area), a modern computer can perform such calls in the order of about five millions
   per second.

   However, consider the nature of falcon as a Virtual Machine interpreted scripting
   language. Creating a complete log message may be an heavy operation by itself;
   for example:
   @code
      load logging

      rendered = ""
      for person in visitors
          rendered += person
          formiddle: rendered += ", "
      end

      GenericLog.log( LOGI, "Today we had " + visitors.len()
          + " visitors, named " + rendered )
   @endcode

   In this case, creating the "rendered" log-friendly representation of the visitors
   is quite slow.

   If there isn't any channel registered with the GenericLog area, the message will
   be discarded, but the heaviest part of the job has already be done, and in waste.

   In case logs are heavy and frequent, it is useful to wrap log generation of
   the most intensive entries in runtime checks, or better, compile time directives
   that prevent calling logs when the application knows they are actually not wanted
   in the current context.

   The @a LogArea.minlog method returns the minimum log level that is accepted
   by a registered channel (the @a gminlog function operates on the GeneralLog area),
   providing a simple way to prevent logging useless data at runtime:

   @code
      load logging
      
      // want debug?
      if LOGD0 <= gminlog()
         // prepare an heavy log...
         logd0( "Debug heavy log: " + ... )
      end
   @endcode

   @note Log levels appares in inverse order of severity; so LOGC (fatal) is a level
   numerically less than LOGD0 (debug).

   @section feather_logging_extending Extending classes.

   All the classes in this module are highly reflective, and operate on the inner
   C++ classes for performance reason. Because of this, overloading a LogChannel
   class and overloading its @b log method won't cause the LogArea to call back
   the new log method provided by the script; the underlying C++ log method will
   still be called.

   However, it is legal to extend the LogChannel classes; the overloaded log
   method can be used directly by the script even if it will be ignored by
   LogArea instances.

   @section feather_logging_engine Integration with the engine logging facility

   Falcon provides a simple logging facility that is meant to be used by the
   command line interpreter or by various environment controllers (WOPI in web
   servers, embedding applications, etc).

   A direct interface to this logging media is provided by the @a Log class in the
   core module.

   However, the engine facility can be transparently used by the server-grade logging
   facility provided by this module through the @a LogChannelEngine class. The engine
   logging facility is seen as a channel to which the log messages that are sent to subscribed
   areas get marshaled.
*/


/*#
   @beginmodule logging
*/

/*# @object GeneralLog
      @from LogArea
      @brief General logging area.

      This is the default log area used by the @a glog function.
  */

#ifndef FALCON_STATIC_FEATHERS

FALCON_MODULE_DECL
{
   Falcon::Module *self = new Falcon::Feathers::ModuleLogging;
   return self;
}

#endif

/* end of logging.cpp */


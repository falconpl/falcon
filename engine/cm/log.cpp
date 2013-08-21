/*
   FALCON - The Falcon Programming Language.
   FILE: log.cpp

   Falcon core module -- Log
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 21 Aug 2013 14:15:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "falcon/cm/log.cpp"

#include <falcon/cm/log.h>

#include <falcon/vm.h>
#include <falcon/log.h>
#include <falcon/vmcontext.h>
#include <falcon/uri.h>
#include <falcon/path.h>
#include <falcon/datawriter.h>
#include <falcon/datareader.h>
#include <falcon/itemarray.h>
#include <falcon/function.h>

#include <falcon/stderrors.h>

namespace Falcon {
namespace Ext {
/*#
 @beginmodule core
 */

/*#
 @class Log
 @brief Interface to engine logger facility.

 This class exposes several static methods that provide access to the standard
 Falcon engine logging facility.

 The facility is used by host applications to receive log messages from several
 sources; one of this source is the script being run in a particular VM. Other
 sources may be elements of the engine itself, parts of the host application
 in need of communicating or third party modules.

 The standard Falcon command line interpreter is able to redirect the log messages
 to a static file or to a system stream. The WOPI modules redirect the log messages
 to the logging facility of the host application (for instance, the standard logs
 of the Apache web server). Different host applications may intercept the log messages
 and manage them as they prefer.

 The internal log engine provides an asynchronous, memory effective and high performance
 message dispatching system. As soon as the message is queued for delivery, the invoking
 context is free to proceed; memory for the messages is retrieved from a pool, where the original
 message string is copied, so that it can be disposed by the calling process at will.

 @note The feather module Log provides a more extensive facility to allow scripts to create
 sever-grade robust logging strategies. It also provides an interface to the internal engine
 log system, as one of the channels that a script can control.

 */
namespace CLog {

static void internal_log( VMContext* ctx, int lvl )
{
   Item* i_msg = ctx->param(0);
   if( i_msg == 0 || ! i_msg->isString() )
   {
      throw ctx->currentFrame().m_function->paramError(__LINE__, SRC);
   }
   String &msg = *i_msg->asString();
   Engine::instance()->log()->log( Log::fac_script, lvl, msg );

   ctx->returnFrame();
}

/*#
 @method c Log
 @brief Logs a message with critical (highest) severity.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( c, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(c)
{
   internal_log( ctx, Log::lvl_critical );
}


/*#
 @method e Log
 @brief Logs a message with error (high) severity.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( e, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(e)
{
   internal_log( ctx, Log::lvl_error );
}

/*#
 @method w Log
 @brief Logs a message with warn (mid) severity.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( w, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(w)
{
   internal_log( ctx, Log::lvl_warn );
}

/*#
 @method i Log
 @brief Logs a message with info (low) severity.
 @param msg The message to be sent.

*/

FALCON_DECLARE_FUNCTION( i, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(i)
{
   internal_log( ctx, Log::lvl_info );
}

/*#
 @method d Log
 @brief Logs a message with detail (lowest) severity.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( d, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(d)
{
   internal_log( ctx, Log::lvl_detail );
}

/*#
 @method d0 Log
 @brief Logs a message intended for debugging.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( d0, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(d0)
{
   internal_log( ctx, Log::lvl_debug );
}

/*#
 @method d1 Log
 @brief Logs a message intended for fine debugging.
 @param msg The message to be sent.

*/

FALCON_DECLARE_FUNCTION( d1, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(d1)
{
   internal_log( ctx, Log::lvl_debug1 );
}

/*#
 @method d2 Log
 @brief Logs a message intended for low level debugging.
 @param msg The message to be sent.

*/
FALCON_DECLARE_FUNCTION( d2, "msg:S" );
FALCON_DEFINE_FUNCTION_P1(d2)
{
   internal_log( ctx, Log::lvl_debug2 );
}

/*#
 @method log Log
 @brief Logs a message with arbitrary severity.
 @param lvl The message severity.
 @param msg The message to be sent.

 Severity can be one of the following class-wide constants:
 - CRIT
 - ERR
 - WARN
 - INFO
 - DET
 - DBG
 - DBG1
 - DBG2

*/
FALCON_DECLARE_FUNCTION( log, "lvl:i, msg:S" );
FALCON_DEFINE_FUNCTION_P1(log)
{
   Item* i_lvl = ctx->param(0);
   Item* i_msg = ctx->param(1);
   if( i_lvl == 0 || ! i_lvl->isOrdinal()
       || i_msg == 0 || ! i_msg->isString() )
   {
      throw ctx->currentFrame().m_function->paramError(__LINE__, SRC);
   }

   int32 lvl = (int32) i_lvl->asInteger();
   String &msg = *i_msg->asString();
   Engine::instance()->log()->log( Log::fac_script, lvl, msg );

   ctx->returnFrame();
}

}

ClassLog::ClassLog():
   Class("Log")
{
   addMethod( new CLog::Function_c, true );
   addMethod( new CLog::Function_e, true );
   addMethod( new CLog::Function_w, true );
   addMethod( new CLog::Function_i, true );
   addMethod( new CLog::Function_d, true );
   addMethod( new CLog::Function_d0, true );
   addMethod( new CLog::Function_d1, true );
   addMethod( new CLog::Function_d2, true );

   addMethod( new CLog::Function_log, true );

   addConstant("CRIT", Log::lvl_critical );
   addConstant("ERR", Log::lvl_error );
   addConstant("WARN", Log::lvl_warn );
   addConstant("INFO", Log::lvl_info );
   addConstant("DET", Log::lvl_detail );
   addConstant("DBG", Log::lvl_debug );
   addConstant("DBG1", Log::lvl_debug1 );
   addConstant("DBG2", Log::lvl_debug2 );
}


ClassLog::~ClassLog()
{}


void ClassLog::dispose( void* ) const
{
}

void* ClassLog::clone( void* ) const
{
   return 0;
}


void* ClassLog::createInstance() const
{
   return 0;
}

}
}

/* end of iterator.cpp */

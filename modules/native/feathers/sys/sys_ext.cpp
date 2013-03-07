/* FALCON - The Falcon Programming Language.
 * FILE: rnd_ext.cpp
 * 
 * Interface to the virtual machine
 * Main module file, providing the module object to the Falcon engine.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai
 * Begin: Wed, 06 Mar 2013 17:24:56 +0100
 * 
 * -------------------------------------------------------------------
 * (C) Copyright 2013: The above AUTHOR
 * 
 * See LICENSE file for licensing details.
 */

#undef SRC
#define SRC "modules/native/feathers/sys/sys_ext.cpp"

/*#
   @beginmodule feathers.sys
*/

#include <falcon/engine.h>
#include <falcon/vmcontext.h>
#include <falcon/stdhandlers.h>
#include <falcon/itemdict.h>

#include <falcon/error.h>
#include <falcon/errors/accesstypeerror.h>

#include <falcon/sys.h>
#include <falcon/processor.h>
#include <falcon/stream.h>
#include <falcon/stdstreams.h>

#include "sys_ext.h"

namespace Falcon { 
    namespace Ext {
       /*#
          @function stdIn
          @brief Access to raw, non transcoded system process input stream.
          @return The raw process input stream.
        */
       void Function_stdIn::invoke( VMContext* ctx, int32 )
       {
          static Stream* stream = new StdInStream(false);
          // we don't give it to the GC; instead, we let the user close it explicitly.
          ctx->returnFrame( Item(stream->handler(), stream) );
       }

       /*#
          @function stdOut
          @brief Access to raw, non transcoded system process output stream.
          @return The raw process output stream.
        */
       void Function_stdOut::invoke( VMContext* ctx, int32 )
       {
          static Stream* stream = new StdOutStream(false);
          // we don't give it to the GC; instead, we let the user close it explicitly.
          ctx->returnFrame( Item(stream->handler(), stream) );
       }

       /*#
          @function stdErr
          @brief Access to raw, non transcoded system process error stream.
          @return The raw process error stream.
        */
       void Function_stdErr::invoke( VMContext* ctx, int32 )
       {
          static Stream* stream = new StdErrStream(false);
          // we don't give it to the GC; instead, we let the user close it explicitly.
          ctx->returnFrame( Item(stream->handler(), stream) );
       }

       /*#
          @function getEnv
          @brief Gets an environment string.
          @param variable The environment variable to be fetched.
          @return The value of the environment variable, or nil if the string doesn't exist.
        */
       void Function_getEnv::invoke( VMContext* ctx, int32 )
       {
          Item* i_varName = ctx->param(0);
          if( i_varName == 0 || ! i_varName->isString() )
          {
             throw paramError(__LINE__, SRC);
          }

          String* result = new String;
          String& var = *i_varName->asString();
          if( ::Falcon::Sys::_getEnv(var, *result) )
          {
             ctx->returnFrame( FALCON_GC_HANDLE(result));
          }
          else {
             delete result;
             ctx->returnFrame();
          }
       }

       /*#
          @function unsetEnv
          @brief Gets an environment string.
          @param variable The environment variable to be set.
          @param value The value (string) to be set.

          If the @b value is nil, on those system where it is
          allowed, the string is removed from the environment;
          in the others, it is simply set to an empty string value.
        */
       void Function_setEnv::invoke( VMContext* ctx, int32 )
       {
          Item* i_varName = ctx->param(0);
          Item* i_value = ctx->param(1);
          if( i_varName == 0 || ! i_varName->isString()
            || i_value == 0 || !(i_value->isString() || i_value->isNil() ) )
          {
             throw paramError(__LINE__, SRC);
          }

          String& var = *i_varName->asString();
          if( i_value->isNil() )
          {
             ::Falcon::Sys::_unsetEnv(var);
          }
          else
          {
             String& value = *i_value->asString();
             ::Falcon::Sys::_setEnv(var, value);
          }

          ctx->returnFrame();
       }

       static void env_callback( const String& key, const String& value, void* cbData )
       {
          ItemDict* dict = static_cast<ItemDict*>(cbData);
          String* k = new String( key );
          k->bufferize();
          String* v = new String( value );
          v->bufferize();
          dict->insert( FALCON_GC_HANDLE(k), FALCON_GC_HANDLE(v) );
       }

       /*#
           @function environ
           @brief Returns all the environment strings in a dictionary.
           @return A dictionary containing all the environment strings and
           their value.
         */
        void Function_environ::invoke( VMContext* ctx, int32 )
        {
           ItemDict* result = new ItemDict;
           ::Falcon::Sys::_enumerateEnvironment( &env_callback, result);
           ctx->returnFrame( FALCON_GC_HANDLE(result) );
        }

       /*#
          @function edesc
          @brief Describe a system error.
          @param ecode Numeric identifier of the system error.
          @return A string containing a (possibly localized) error description.
        */
       void Function_edesc::invoke( VMContext* ctx, int32 )
       {
          Item* i_ecode = ctx->param(0);
          if( i_ecode == 0 || ! i_ecode->isOrdinal() )
          {
             throw paramError(__LINE__, SRC);
          }

          int64 ecode = i_ecode->forceInteger();
          String* desc = new String;
          if( ! ::Falcon::Sys::_describeError(ecode, *desc) )
          {
             desc->append("?");
          }
          ctx->returnFrame(FALCON_GC_HANDLE(desc));
       }


       /*#
          @function cores
          @brief Return the count of detected cores (or CPUs) in the system.
          @return number of detected cores or 0 if it wasn't possible to detect them.
        */
       void Function_cores::invoke( VMContext* ctx, int32 )
       {
          ctx->returnFrame(::Falcon::Sys::_getCores());
       }

       /*#
          @function epoch
          @brief Access to raw, non transcoded system process input stream.
        */
       void Function_epoch::invoke( VMContext* ctx, int32 )
       {
          ctx->returnFrame(::Falcon::Sys::_epoch());
       }

       /*#
          @function systemType
          @brief Returns an overall description of the system type.
          @return "WIN" or "UNIC"
        */
       void Function_systemType::invoke( VMContext* ctx, int32 )
       {
          ctx->returnFrame(FALCON_GC_HANDLE((new String( FALCON_HOST_SYSTEM ))));
       }

    }
} // namespace Falcon::Ext

/* end of sys_ext.cpp */


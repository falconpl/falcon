/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.cpp
   $Id: process_ext.cpp,v 1.10 2007/08/11 00:11:56 jonnymind Exp $

   Process module -- Falcon interface functions

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
   Process module -- Falcon interface functions
   This is the module implementation file.
*/

#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include "process_sys.h"
#include "process_mod.h"
#include "process_ext.h"

namespace Falcon {
namespace Ext {

/**
   processId() --> ID of the current process
*/

FALCON_FUNC  falcon_processId( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) Sys::processId() );
}

/**
   processKill( pid, [severe] ) --> result
*/

FALCON_FUNC  falcon_processKill( ::Falcon::VMachine *vm )
{
   Item *id = vm->param(0);
   Item *mode = vm->param(1);

   if ( id == 0 || ! id->isOrdinal() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
   }

   if ( mode == 0 || ! mode->isTrue() )
   {
      vm->retval( (int64) (Sys::processTerminate( id->forceInteger() ) ? 1:0 ) );
   }
   else {
      vm->retval( (int64) (Sys::processKill( id->forceInteger() ) ? 1:0 ) );
   }
}

FALCON_FUNC  ProcessEnum_init  ( ::Falcon::VMachine *vm )
{
   Sys::ProcessEnum *pe = new Sys::ProcessEnum();
   CoreObject *self = vm->self().asObject();
   self->setUserData( pe );
}

FALCON_FUNC  ProcessEnum_next  ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ProcessEnum *pe = (Sys::ProcessEnum *)self->getUserData();
   GarbageString *name = new GarbageString( vm );
   GarbageString *path = new GarbageString( vm );
   uint64 pid, ppid;

   int64 res = (int64) pe->next( *name, pid, ppid, *path );

   if ( res != 1 )
   {
      vm->memPool()->destroyGarbage( name );
      vm->memPool()->destroyGarbage( path );

      if ( res == -1 )
      {
         vm->raiseModError( new ProcessError( ErrorParam( 1021, __LINE__ ).
            extra( "Error while reading the process list" ) ) );
         return;
      }
   }
   else {
      self->setProperty( "name", name );
      self->setProperty( "cmdLine", path );
      self->setProperty( "pid", (int64) pid );
      self->setProperty( "parentPid", (int64) ppid );
   }

   vm->retval(res);
}

FALCON_FUNC  ProcessEnum_close  ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ProcessEnum *pe = (Sys::ProcessEnum *)self->getUserData();
   if ( ! pe->close() ) {
         vm->raiseModError( new ProcessError( ErrorParam( 1022, __LINE__ ).
            extra( "Error while closing the process list" ) ) );
      return;
   }
}


/**
   system( string, [waitMode] ) --> int
      waitMode = TRUE - foreground
      waitMode = FALSE - background
*/
FALCON_FUNC  falcon_system ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode = vm->param(1);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   bool background = mode == 0 ? false : mode->isTrue();
   String *argv[4];

   String shellName( ::Falcon::Sys::shellName() );
   String shellParam( ::Falcon::Sys::shellParam() );
   argv[0] = &shellName;
   argv[1] = &shellParam;
   argv[2] = sys_req->asString();
   argv[3] = 0;

   int retval;
   if( ::Falcon::Sys::spawn( argv, false, background, &retval ) )
      vm->retval( retval );
   else {
      vm->raiseModError( new ProcessError( ErrorParam( 1022, __LINE__ ).
         extra( "Error while closing the process list" ).sysError( retval ) ) );
   }
}

/**
   systemCall( string, [waitMode] ) --> int
      waitMode = TRUE - foreground
      waitMode = FALSE - background
*/
FALCON_FUNC  falcon_systemCall ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode = vm->param(1);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING &&  sys_req->type() != FLC_ITEM_ARRAY ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   bool background = mode == 0 ? false : mode->isTrue();
   String **argv;

   if( sys_req->type() == FLC_ITEM_STRING ) {
      argv = ::Falcon::Mod::argvize( *sys_req->asString(), false );
   }
   else {
      uint32 count;
      CoreArray *array = sys_req->asArray();
      for( count = 0; count < array->length(); count ++ )
         if ( array->at( count ).type() != FLC_ITEM_STRING ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
               extra( "All the elements in the first parameter must be strings" ) ) );
            return;
         }

      argv = (String **) memAlloc( (array->length()+1) * sizeof( String * ) );
      for( count = 0; count < array->length(); count ++ )
      {
         argv[count] = (*array)[count].asString();
      }
      argv[array->length()] = 0;
   }

   int retval;
   if( ::Falcon::Sys::spawn( argv, false, background, &retval ) )
      vm->retval( retval );
   else {
      vm->raiseModError( new ProcessError( ErrorParam( 1020, __LINE__ ).
         desc( "Can't open the process" ).sysError( retval ) ) );
   }

   if( sys_req->type() == FLC_ITEM_STRING )
      ::Falcon::Mod::freeArgv( argv );
   else
      memFree( argv );
}

/**
   exec( string )
   exec( array )
*/
FALCON_FUNC  falcon_exec ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING &&  sys_req->type() != FLC_ITEM_ARRAY ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ) ) );
      return;
   }

   String **argv;

   if( sys_req->type() == FLC_ITEM_STRING ) {
      argv = ::Falcon::Mod::argvize( *sys_req->asString(), false );
   }
   else {
      uint32 count;
      CoreArray *array = sys_req->asArray();
      for( count = 0; count < array->length(); count ++ )
         if ( array->at( count ).type() != FLC_ITEM_STRING ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
               extra( "All the elements in the first parameter must be strings" ) ) );
            return;
         }

      argv = (String **) memAlloc( (array->length()+1) * sizeof( char * ) );

      for( count = 0; count < array->length(); count ++ )
      {
         argv[count] = array->at( count ).asString();
      }
      argv[array->length()] = 0;
   }

   int retval;
   if( ::Falcon::Sys::spawn( argv, true, false, &retval ) )
      vm->retval( retval );
   else {
      vm->raiseModError( new ProcessError( ErrorParam( 1020, __LINE__ ).
         desc( "Can't open the process" ).sysError( retval ) ) );
   }

   if( sys_req->type() == FLC_ITEM_STRING )
      ::Falcon::Mod::freeArgv( argv );
   else
      memFree( argv );
}


/**
   procHandle = Process._init( [name|array], attributes )

   attributes:
      0x1 - sink in
      0x2 - sink out
      0x4 - sink err
      0x8 - merge out and error streams
      0x10 - background
      0x20 - use system shell to execute command.
*/

FALCON_FUNC  Process_init ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode_itm = vm->param(1);

   if( sys_req == 0 || ( ! sys_req->isString() && ! sys_req->isArray() ) ||
      (mode_itm != 0 && ! mode_itm->isOrdinal())  )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "( S|A, [I] )" ) ) );
      return;
   }

   String **argv;
   String *args[4];
   bool delArgs;
   bool deepDel;
   // this will also work as flag, as it is valorized only when using the static args[] vector.
   uint32 mode = mode_itm == 0 ? 0 : (uint32) mode_itm->forceInteger();

   //pa_viaShell
   String shellName( ::Falcon::Sys::shellName() );
   String shellParam( ::Falcon::Sys::shellParam() );
   if ( (mode & 0x20) == 0x20 && sys_req->type() == FLC_ITEM_STRING ) {
      delArgs = false;
      argv = args;
      argv[0] = &shellName;
      argv[1] = &shellParam;
      argv[2] = sys_req->asString();
      argv[3] = 0;
   }
   else if( sys_req->type() == FLC_ITEM_STRING ) {
      delArgs = true;
      deepDel = true;
      argv = ::Falcon::Mod::argvize( *sys_req->asString(), false );
   }
   else {
      delArgs = true;
      deepDel = false;
      uint32 count;
      CoreArray *array = sys_req->asArray();
      for( count = 0; count < array->length(); count ++ )
         if ( array->at( count ).type() != FLC_ITEM_STRING ) {
            vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( "All the elements in the first parameter must be strings" ) ) );
            return;
         }
      argv = (String **) memAlloc( (array->length()+1) * sizeof( String * ) );
      for( count = 0; count < array->length(); count ++ )
      {
         argv[count] = (*array)[count].asString();
      }
      argv[array->length()] = 0;
   }

   bool sinkin = ((mode & 0x1) == 0x1);
   bool sinkout = ((mode & 0x2) == 0x2);
   bool sinkerr = ((mode & 0x4) == 0x4);
   bool mergeerr = ((mode & 0x8) == 0x8);
   bool background = ((mode & 0x10) == 0x10);

   ::Falcon::Sys::ProcessHandle *handle = ::Falcon::Sys::openProcess( argv, sinkin, sinkout, sinkerr, mergeerr, background );
   if ( handle->lastError() == 0 )
      vm->self().asObject()->setUserData( handle );
   else {
      vm->raiseModError( new ProcessError( ErrorParam( 1020, __LINE__ ).
         desc( "Can't open the process" ).sysError( handle->lastError() ) ) );
      delete handle;
   }

   if ( delArgs )
   {
      if( deepDel )
         ::Falcon::Mod::freeArgv( argv );
      else
         memFree( argv );
   }
}

FALCON_FUNC  Process_wait ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();
   if( ! handle->wait( true ) ) {
      vm->raiseModError( new ProcessError( ErrorParam( 1121, __LINE__ ).
         desc( "Wait failed" ).sysError( handle->lastError() ) ) );
   }
}

FALCON_FUNC  Process_close ( ::Falcon::VMachine *vm )
{
   Item *severe = vm->param(0);
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();


   if ( ! handle->done() )
   {
      bool sev = severe == 0 ? false : severe->isTrue();
      if( ! handle->terminate( sev ) ) {
         vm->raiseModError( new ProcessError( ErrorParam( 1122, __LINE__ ).
            desc( "Terminate failed" ).sysError( handle->lastError() ) ) );
      }
      if( ! handle->wait( true ) ) {
         vm->raiseModError( new ProcessError( ErrorParam( 1123, __LINE__ ).
            desc( "Wait failed" ).sysError( handle->lastError() ) ) );
      }
   }
   handle->close();
}

FALCON_FUNC  Process_value ( ::Falcon::VMachine *vm )
{
   Item *h_wait = vm->param(0);

   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
         vm->self().asObject()->getUserData();

   bool wait = h_wait == 0 ? false : h_wait->isTrue();
   if ( wait && ! handle->done() ) {
      if( ! handle->wait( true ) ) {
         vm->raiseModError( new ProcessError( ErrorParam( 1123, __LINE__ ).
            desc( "Wait failed" ).sysError( handle->lastError() ) ) );
      }
   }
   // give a test to see if the process is terminated in the meanwhile
   else if ( ! handle->done() ) {
      if( ! handle->wait( false ) ) {
         vm->raiseModError( new ProcessError( ErrorParam( 1123, __LINE__ ).
            desc( "Wait failed" ).sysError( handle->lastError() ) ) );
      }
   }

   if( handle->done() )
      vm->retval( handle->processValue() );
   else
      vm->retval( -1 ); // not yet terminated.
}


FALCON_FUNC  Process_getInput ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getInputStream();
   if (file == 0 )
      vm->retnil();
   else {
      Item *stream_class = vm->findGlobalItem( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}


FALCON_FUNC  Process_getOutput ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getOutputStream();
   if (file == 0 )
      vm->retnil();
   else{
      Item *stream_class = vm->findGlobalItem( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}

FALCON_FUNC  Process_getAux ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getErrorStream();
   if (file == 0 )
      vm->retnil();
   else {
      Item *stream_class = vm->findGlobalItem( "Stream" );
      //if the rtl, that already returned File service, is right, this can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}

FALCON_FUNC  ProcessError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new Falcon::ErrorCarrier( new ProcessError ) );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of process_mod.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: process_mod.cpp

   Process module -- Falcon interface functions

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Process module -- Falcon interface functions
   This is the module implementation file.
*/
#include <cstdio>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/carray.h>
#include <falcon/memory.h>
#include <falcon/fassert.h>
#include "process_sys.h"
#include "process_mod.h"
#include "process_ext.h"
#include "process_st.h"



/*#
    @beginmodule feather_process
*/

namespace Falcon {
namespace Ext {

/*#
   @function processId
   @brief Returns the process ID of the process hosting the Falcon VM.
   @return a numeric process ID.

   For command line Falcon interpreter, this ID may be considered the ID
   of the Falcon program being executed; in embedding applications, the
   function will return the process ID associated with the host application.
*/

FALCON_FUNC  falcon_processId( ::Falcon::VMachine *vm )
{
   vm->retval( (int64) Sys::processId() );
}

/*#
   @function processKill
   @brief Terminates the given process given its ID, if possible.
   @param pid The Process ID of the process that should be terminated.
   @optparam severe If given and true, use the maximum severity allowable to stop the given process.
   @return True on success, false on failure.

   The process having the given PID is terminated. On UNIX systems,
   a TERM signal is sent to the process. If severe is true, the process
   is stopped in the most hard way the system provides; i.e. in UNIX, KILL
   signal is sent.
*/
FALCON_FUNC  falcon_processKill( ::Falcon::VMachine *vm )
{
   Item *id = vm->param(0);
   Item *mode = vm->param(1);

   if ( id == 0 || ! id->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   if ( mode == 0 || ! mode->isTrue() )
   {
      vm->regA().setBoolean( Sys::processTerminate( id->forceInteger() ) );
   }
   else {
      vm->regA().setBoolean( Sys::processKill( id->forceInteger() ) );
   }
}

/*#
   @class ProcessEnum
   @brief Provides a list of currently executed process.

   This class can be used to retreive a list of running processes
   on the host machine, with minimal
   informations for each of them as its name and an unique ID by which
   it can be identified later on. ProcessEnum constructor returns an object
   that can be iterated upon like in the following example.

   @code
   enum = ProcessEnum()

   while enum.next()
      > enum.name, ":", enum.pid, "( child of ", enum.parentPid, ")"
   end
   @endcode

   The next() method will fill the object properties with the data of a new
   element, until it returns false.

   @prop cmdLine Complete path of the program that started the process.
                 Not always available, and not provided by all systems.
   @prop name Symbolic name of the process.
   @prop pid ID (usually numeric) identifying a process in the system; this value can
      be directly fed as a parameter for function accepting process IDs.
   @prop parentPid ID of the process that created this process. The parent PID always
      represents an existing process, but it's possible that children are
      returned before the parent is listed.
*/

FALCON_FUNC  ProcessEnum_init  ( ::Falcon::VMachine *vm )
{
   Sys::ProcessEnum *pe = new Sys::ProcessEnum();
   CoreObject *self = vm->self().asObject();
   self->setUserData( pe );
}


/*#
   @method next ProcessEnum
   @brief Fills the properties of this class with the next entry in the process table.
   @return True if there is an element that will fill the properties, false if the element
      enumeration terminated.
   @raise ProcessError on system error.

   Fills the properties of the class with data from the next
   element in the process enumeration. If the previous one was the last
   entry, the method returns false, else it returns true and the
   properties are changed.
*/

FALCON_FUNC  ProcessEnum_next  ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ProcessEnum *pe = (Sys::ProcessEnum *)self->getUserData();
   CoreString *name = new CoreString;
   CoreString *path = new CoreString;
   uint64 pid, ppid;

   int64 res = (int64) pe->next( *name, pid, ppid, *path );

   if ( res != 1 )
   {
      if ( res == -1 )
      {
         throw new ProcessError( ErrorParam( FALPROC_ERR_READLIST, __LINE__ )
            .desc( FAL_STR(proc_msg_errlist) ) );
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


/*#
   @method close ProcessEnum
   @brief Closes the enumeration freeing system resources.

   Disposes the data associated with this item without waiting
   for the garbage collector to reclaim them.
*/
FALCON_FUNC  ProcessEnum_close  ( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   Sys::ProcessEnum *pe = (Sys::ProcessEnum *)self->getUserData();
   if ( ! pe->close() ) {
         throw new ProcessError( ErrorParam( FALPROC_ERR_CLOSELIST, __LINE__ )
            .desc( FAL_STR( proc_msg_errlist2 ) ) );
      return;
   }
}


/*#
   @function system
   @brief Executes an external process via command shell, and waits for its termination.
   @param command The command to be executed.
   @param background If given and true, the process runs hidden.
   @return Exit status of the process.
   @raise ProcessError if the process couldn't be created.

   This function launches an external system command and waits until the command
   execution is complete, returning the exit code of the child process. The process
   is actually executed by passing the command string to the system command shell.
   In this way, it is possible to  execute commands that are parsed by the shell.

   This includes internal commands as "dir" in Windows systems, or small scripts as
   "for file in $(ls); do touch $file; done" if the system shell is sh.
   However, loading the shell may generate a needless overhead for the most common
   usages of system(). Use systemCall() if there isn't the need to have the system
   shell to parse the command.

   If the background parameter is true, the execution of the child process is
   hidden; in example, on systems allocating virtual consoles to new processes, the
   child is given the parent's console instead. When running inside Window Managers
   and graphical systems, icons representing the process are usually not visible when
   this option is set.
*/
FALCON_FUNC  falcon_system ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode = vm->param(1);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
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
   vm->idle();
   if( ::Falcon::Sys::spawn( argv, false, background, &retval ) )
   {
      vm->unidle();
      vm->retval( retval );
   }
   else 
   {
      vm->unidle();
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATLIST, __LINE__ )
         .desc( FAL_STR(proc_msg_errlist3) )
         .sysError( retval ) );
   }
}

/*#
   @function systemCall
   @brief Executes an external process and waits for its termination.
   @param command The command to be executed.
   @param background If given and true, the process runs hidden.
   @return Exit status of the process.
   @raise ProcessError if the process couldn't be created.

   This function launches an external system command and waits until the command
   execution is terminated, returning the exit code of the child process. The
   command is searched in the system path, if an absolute path is not given. A
   simple parsing is performed on the string executing the command, so that
   parameters between quotes are sent to the child process as a single parameter;
   in example:

   @code
   retval = systemCall( "myprog alfa beta \"delta gamma\" omega" )
   @endcode

   In this  case, the child process will receive four parameters, the third of
   which being the two words between quotes. Those quotes are not passed to the
   child process.

   If the background parameter is true, the execution of the child process is
   hidden; in example, on systems allocating virtual consoles to new processes, the
   child is given the parent's console instead. Icons representing the process are
   usually not visible when this option is set.
*/
FALCON_FUNC  falcon_systemCall ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode = vm->param(1);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING &&  sys_req->type() != FLC_ITEM_ARRAY ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   vm->idle();
   
   bool background = mode == 0 ? false : mode->isTrue();
   String **argv;

   if( sys_req->isString() ) {
      argv = ::Falcon::Mod::argvize( *sys_req->asString(), false );
   }
   else {
      uint32 count;
      CoreArray *array = sys_req->asArray();
      for( count = 0; count < array->length(); count ++ )
         if ( array->at( count ).type() != FLC_ITEM_STRING ) {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
               extra( FAL_STR( proc_msg_allstr ) ) );
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
   {
      vm->unidle();
      vm->retval( retval );
      
      if( sys_req->type() == FLC_ITEM_STRING )
         ::Falcon::Mod::freeArgv( argv );
      else
         memFree( argv );
   }
   else {
      vm->unidle();
       if( sys_req->type() == FLC_ITEM_STRING )
         ::Falcon::Mod::freeArgv( argv );
      else
         memFree( argv );
         
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
         .desc( FAL_STR( proc_msg_prccreate ) )
         .sysError( retval ) );
   }
   
}


/*#
   @function pread
   @brief Executes an external process and waits for its termination.
   @param command The command to be executed.
   @param background If given and true, the process runs hidden.
   @return The full output generated by the rpcess.
   @raise ProcessError if the process couldn't be created.

   This function launches an external system command and waits until the command
   execution is terminated, returning the exit code of the child process. All the 
   
   for example:

   @code
   dir_contents = pread( "ls" )
   > dir_contents
   @endcode

   If the process cannot be started if it fails to start, an error is raised.
   
   \note This function uses the standard system shell to execute the passed
   commands, so it is possible to pipe applications and redirect streams
   via the standard "|" and ">" command line characters.
*/
FALCON_FUNC  falcon_pread ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode = vm->param(1);

   if( sys_req == 0 || ( !sys_req->isString() &&  !sys_req->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
   }

   bool background = mode == 0 ? false : mode->isTrue();                    
   String shellName( ::Falcon::Sys::shellName() );
   String shellParam( ::Falcon::Sys::shellParam() );
   GenericVector argv(&traits::t_stringptr());
   
   if( sys_req->isString() )
   {
      argv.push( &shellName );
      argv.push( &shellParam );
      argv.push( sys_req->asString() );
      argv.push( 0 );
   }
   else
   {
      CoreArray *array = sys_req->asArray();
      for( size_t i = 0; i < array->length(); i++ )
         if ( !array->at( i ).isString() )
         {
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( proc_msg_allstr ) ) );
         }

      for( size_t i = 0; i < array->length(); i++ )
         argv.push( (*array)[i].asString() );
       argv.push( 0 );
   }
 
   int retval = 0;
   CoreString* gs = new CoreString;
   if( ::Falcon::Sys::spawn_read( static_cast<String**>( argv.at(0) ),
                                  false, background, &retval, gs ) )
   {
      if ( retval == 0x7F00 )
      {
         throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
            .desc( FAL_STR( proc_msg_prccreate ) )
            .sysError( 0 ) );
      }

      vm->retval( gs );
   }
   else
   {
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
         .desc( FAL_STR( proc_msg_prccreate ) )
         .sysError( retval ) );
   }

}

/*#
   @function exec
   @brief Launches a process in place of the host process.
   @param command A single string with the system command, or an array of parameters.
   @return Never returns.
   @raise ProcessError if the process couldn't be created

   On Unix-like systems, and wherever this feature is present, exec() calls an
   underlying OS request that swaps out the host process and executes the required
   command. This feature is often desirable for scripts that has just to setup an
   environment for a program they call thereafter, or for scripts selecting one
   program to execute from a list of possible programs. Where this function call is
   not provided, exec is implemented by calling the required process passing the
   current environment and then quitting the host program in the fastest possible
   way.

   Embedders may be willing to turn off this feature by unexporting the exec symbol
   before linking the process module in the VM.

   The @b command parameter may be a complete string holding the command line of the
   program to be executed, or it may be an array where the first element is the
   program, and the other elements will be passed as parameters. If the program
   name is not an absolute path, it is searched in the system execution search path.
*/
FALCON_FUNC  falcon_exec ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);

   if( sys_req == 0 || ( sys_req->type() != FLC_ITEM_STRING &&  sys_req->type() != FLC_ITEM_ARRAY ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ) );
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
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
               extra( FAL_STR( proc_msg_allstr ) ) );
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
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ ).
         desc( FAL_STR( proc_msg_prccreate ) ).sysError( retval ) );
   }

   if( sys_req->type() == FLC_ITEM_STRING )
      ::Falcon::Mod::freeArgv( argv );
   else
      memFree( argv );
}


/*#
   @class Process
   @brief Execute and control child processes.
   @param command A string representing the program to be executed
        and its arguments, or an array whose first element is the program, and
        the others are the arguments.
   @optparam flags process open flags.

   This class is meant for finer control of child processes and
   inter process comunication.
   
   The process named in the @b command argument is started. It is possible to
   provide either a string containing a complete command line, with the process
   name and its arguments, or an array whose first element is the process name,
   and the other elements are the parameters that will be provided to the
   process.

   The optional @b flags parameter can control the behavior of the started process,
   and may be a combination of the followings:

   - PROCESS_SINK_INPUT: prevent the child process to wait for input from us.
   - PROCESS_SINK_OUTPUT: destroy all the child process output.
   - PROCESS_SINK_AUX: destroy all the child process auxiliary stream output.
   - PROCESS_MERGE_AUX: merge output and auxiliary stream so that they are read
                        by just reading the output stream.
   - PROCESS_BG: Put the process in background/hidden mode.
   - PROCESS_USE_SHELL: Use host system shell to launch the process or execute the command.

*/

FALCON_FUNC  Process_init ( ::Falcon::VMachine *vm )
{
   Item *sys_req = vm->param(0);
   Item *mode_itm = vm->param(1);

   if( sys_req == 0 || ( ! sys_req->isString() && ! sys_req->isArray() ) ||
      (mode_itm != 0 && ! mode_itm->isOrdinal())  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "( S|A, [I] )" ) );
   }

   String **argv;
   String *args[4];
   bool delArgs = false;
   bool deepDel = false;
   // this will also work as flag, as it is valorized only when using the static args[] vector.
   uint32 mode = mode_itm == 0 ? 0 : (uint32) mode_itm->forceInteger();

   //pa_viaShell
   String shellName( ::Falcon::Sys::shellName() );
   String shellParam( ::Falcon::Sys::shellParam() );
   if ( (mode & 0x20) == 0x20 && sys_req->isString() ) {
      delArgs = false;
      argv = args;
      argv[0] = &shellName;
      argv[1] = &shellParam;
      argv[2] = sys_req->asString();
      argv[3] = 0;
   }
   else if( sys_req->isString() ) {
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
            throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
               .extra( FAL_STR( proc_msg_allstr ) ) );
         }
      argv = new String*[array->length() + 1];
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
      delete handle;
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
         .desc( FAL_STR(proc_msg_prccreate) )
         .sysError( handle->lastError() ) );
   }

   if ( delArgs )
   {
      if( deepDel )
         ::Falcon::Mod::freeArgv( argv );
      else
         memFree( argv );
   }
}


/*#
   @method wait Process
   @brief Waits for a child process to terminate.
   @raise ProcessError on system errors or wait failed.

   Waits for the child process to terminate cleanly.

   The thread in which the VM runs will be blocked until the child
   process terminates its execution. After this call, the script should
   also call the @a Process.value method to free the system data associated with
   the child process. Use the Processs.value method to test periodically for
   the child process to be completed while the Falcon program continues
   its execution.

   @note At the moment this function doesn't respect the VM interruption
   protocol, but this feature shall be introduced shortly. Until this
   feature is available, the @a Process.value method can be used
   to check if the child process terminated at time intervals.
*/

FALCON_FUNC  Process_wait ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();
   
   vm->idle();
   if( ! handle->wait( true ) ) 
   {
      vm->unidle();
      throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
         .desc( FAL_STR( proc_msg_waitfail ) )
         .sysError( handle->lastError() ) );
   }
   else {
      handle->close();
      vm->unidle();
   }
}

/*#
   @method terminate Process
   @brief Terminate a child process.
   @optparam severe If given and true, use the maximum severity.
   @raise ProcessError on system error.

   Terminates the child process, sending it a request to exit as soon as possible.
   The call returns immediately; it is then necessary to wait for the process
   to actually exit and free its resources through @a Process.value.

   If the @b severe
   parameter is true, then the maximum severity allowed for the host system is
   used. On UNIX, a KILL signal is sent to the child process, while a TERM signal
   is sent if severe is not specified or false.
*/
FALCON_FUNC  Process_terminate ( ::Falcon::VMachine *vm )
{
   Item *severe = vm->param(0);
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();


   if ( ! handle->done() )
   {
      bool sev = severe == 0 ? false : severe->isTrue();
      if( ! handle->terminate( sev ) ) {
         throw new ProcessError( ErrorParam( FALPROC_ERR_TERM, __LINE__ ).
            desc( FAL_STR( proc_msg_termfail ) ).sysError( handle->lastError() ) );
      }
   }
}


/*#
   @method value Process
   @brief Retreives exit value of the child process (and close its handles).
   @optparam wait if given and true, the wait for the child process to be completed.
   @return On success, the exit value of the child process, or -1 if the child process
      is not yet terminated.
   @raise ProcessError on system error.

   Checks whether the child process has completed its execution, eventually
   returning its exit code. If the process is still active, nil will be returned.
   If a true value is provided as parameter, the function will block the VM
   execution until the child process is completed.

   After value() returns, there may still be some data to be read from the child
   output and auxiliary streams; they should be read until they return 0.
*/
FALCON_FUNC  Process_value ( ::Falcon::VMachine *vm )
{
   Item *h_wait = vm->param(0);

   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
         vm->self().asObject()->getUserData();

   bool wait = h_wait == 0 ? false : h_wait->isTrue();
   if ( wait && ! handle->done() ) {
      vm->idle();
      if( ! handle->wait( true ) ) {
         handle->close();
         vm->unidle();
         throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
            .desc( FAL_STR( proc_msg_waitfail ) )
            .sysError( handle->lastError() ) );
      }
      else
         vm->unidle();
   }
   // give a test to see if the process is terminated in the meanwhile
   else if ( ! handle->done() ) {
      if( ! handle->wait( false ) ) {
         throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
            .desc( FAL_STR( proc_msg_waitfail ) )
            .sysError( handle->lastError() ) );
      }
   }

   if( handle->done() )
   {
      vm->retval( handle->processValue() );
      handle->close();
   }
   else
      vm->retval( -1 ); // not yet terminated.
}

/*#
   @method getInput Process
   @brief Returns the process input stream.
   @return The child process input stream (write-only)

   The returned stream can be used as a Falcon stream,
   but it supports only write operations.

   If the process has been opened with the PROCESS_SINK_IN,
   the function will return nil.

   @note This function should be called only once per Process class;
      be sure to cache its value.
*/
FALCON_FUNC  Process_getInput ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getInputStream();
   if (file == 0 )
      vm->retnil();
   else {
      Item *stream_class = vm->findWKI( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}

/*#
   @method getOutput Process
   @brief Returns the process output stream.
   @return The child process output stream (read-only)

   The returned stream can be used as a Falcon stream,
   but it supports only read operations.

   If the process has been opened with the PROCESS_SINK_OUTPUT flag,
   the function will return nil.

   @note This function should be called only once per Process class;
      be sure to cache its value.
*/
FALCON_FUNC  Process_getOutput ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getOutputStream();
   if (file == 0 )
      vm->retnil();
   else{
      Item *stream_class = vm->findWKI( "Stream" );
      //if we wrote the std module, can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}

/*#
   @method getAux Process
   @brief Returns the process auxiliary output stream.
   @return The child process auxiliary output stream (read-only)

   The returned stream can be used as a Falcon stream,
   but it supports only read operations.

   If the process has been opened with the PROCESS_SINK_AUX or
   PROCESS_MERGE_AUX, this method will return nil. In the latter
   case, all the output that should usually go into this stream
   will be sent to the output stream, and it will be possible to
   read it from the stream handle returned by @a Process.getOutput.

   @note This function should be called only once per Process class;
      be sure to cache its value.
*/
FALCON_FUNC  Process_getAux ( ::Falcon::VMachine *vm )
{
   ::Falcon::Sys::ProcessHandle *handle = (::Falcon::Sys::ProcessHandle *)
      vm->self().asObject()->getUserData();

   Stream *file = handle->getErrorStream();
   if (file == 0 )
      vm->retnil();
   else {
      Item *stream_class = vm->findWKI( "Stream" );
      //if the rtl, that already returned File service, is right, this can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}


/*#
   @class ProcessError
   @brief Error generated by process related system failures.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   See the Error class in the core module.
*/

FALCON_FUNC  ProcessError_init ( ::Falcon::VMachine *vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new ProcessError );

   ::Falcon::core::Error_init( vm );
}

}
}

/* end of process_mod.cpp */

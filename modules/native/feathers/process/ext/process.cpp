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
#include "../sys/process.h"
#include "../mod/process.h"
#include "process.h"
#include "../process_st.h"



/*#
    @beginmodule feathers.process
*/

namespace Falcon { namespace Ext {


namespace {

bool s_checkArray(Item* item)
{
   fassert( item->isArray() );

   bool doThrow = false;
   CoreArray *array = item->asArray();
   if ( !( array->length() > 1) )
      return false;

   for( size_t i = 0; i < array->length(); i++ )
      if ( !array->at( i ).isString() )
         return false;

   return true;
}

void s_appendCommands(GenericVector& argv, Item* command)
{
   fassert( s_checkArray(command) );

   CoreArray* commands = command->asArray();
   for( size_t i = 0; i < commands->length(); i++ )
   {
      String* str =  (*commands)[i].asString();
      argv.push( new String( *str ) );
   }
}

String s_mergeCommandArray(Item* command)
{
   fassert( s_checkArray(command) );

   String ret;

   CoreArray* commands = command->asArray();
   ret.append( *(*commands)[0].asString() );
   for( size_t i = 1; i < commands->length(); i++ )
   {
      String* str =  (*commands)[i].asString();
      ret.append( " "  );
      ret.append( *str  );
   }

   return ret;
}

} // anonymous namespace


/*#
   @function processId
   @brief Returns the process ID of the process hosting the Falcon VM.
   @return a numeric process ID.

   For command line Falcon interpreter, this ID may be considered the ID
   of the Falcon program being executed; in embedding applications, the
   function will return the process ID associated with the host application.
*/

FALCON_FUNC  process_processId( VMachine* vm )
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
FALCON_FUNC  process_processKill( VMachine* vm )
{
   Item *id = vm->param(0);
   Item *mode = vm->param(1);

   if ( id == 0 || ! id->isOrdinal() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                            .extra("I, [B]"));
   }

   if ( mode == 0 || ! mode->isTrue() )
   {
      vm->retval( (bool) Sys::processTerminate( id->forceInteger() ) );
   }
   else {
      vm->retval( (bool) Sys::processKill( id->forceInteger() ) );
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

FALCON_FUNC  ProcessEnum::init( VMachine* vm ) { }

CoreObject* ProcessEnum::factory(const CoreClass* cls, void* user_data, bool )
{
   return new Mod::ProcessEnum(cls);
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

FALCON_FUNC  ProcessEnum::next( VMachine* vm )
{
   Mod::ProcessEnum* self = Falcon::dyncast<Mod::ProcessEnum*>( vm->self().asObject() );
   CoreString *name = new CoreString;
   CoreString *commandLine = new CoreString;
   uint64 pid, ppid;

   int64 res = (int64) self->handle()->next( *name, pid, ppid, *commandLine );

   if ( res != 1 )
   {
      if ( res == -1 )
      {
         throw new ProcessError( ErrorParam( FALPROC_ERR_READLIST, __LINE__ )
            .desc( FAL_STR(proc_msg_errlist) ) );
      }
   }
   else
   {
      self->setProperty( "name", name );
      self->setProperty( "cmdLine", commandLine );
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
FALCON_FUNC  ProcessEnum::close( VMachine* vm )
{
   Mod::ProcessEnum* self = Falcon::dyncast<Mod::ProcessEnum*>( vm->self().asObject() );
   if ( ! self->handle()->close() )
   {
      throw new ProcessError( ErrorParam( FALPROC_ERR_CLOSELIST, __LINE__ )
            .desc( FAL_STR( proc_msg_errlist2 ) ) );
   }
}

void ProcessEnum::registerExtensions( Module* self )
{
   Falcon::Symbol *pe_class = self->addClass( "ProcessEnum", ProcessEnum::init );
   pe_class->getClassDef()->factory(&ProcessEnum::factory);
   self->addClassProperty( pe_class, "name" );
   self->addClassProperty( pe_class, "pid" );
   self->addClassProperty( pe_class, "parentPid" );
   self->addClassProperty( pe_class, "cmdLine" );

   self->addClassMethod( pe_class, "next", ProcessEnum::next );
   self->addClassMethod( pe_class, "close", ProcessEnum::close );
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
FALCON_FUNC  process_system( VMachine* vm )
{
   Item *command = vm->param(0);
   Item *mode = vm->param(1);

   if( command == 0 || (!command->isString() && !command->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                            .extra("S|A{S}, [B]") );
   }

   bool background = mode == 0 ? false : mode->isTrue();
   GenericVector argv( &traits::t_stringptr_own() );

   argv.push( new String( Sys::shellName()) );
   argv.push( new String( Sys::shellParam()) );
   if( command->isString() )
      argv.push( new String( *command->asString() ) );
   else
   {
      if ( !s_checkArray(command) )
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                               .extra( FAL_STR( proc_msg_allstr ) ) );
      argv.push( new String( s_mergeCommandArray(command) ) );
   }
   argv.push( 0 );

   int retval;
   vm->idle();
   if( Sys::spawn( static_cast<String**>( argv.at(0) ),
                             false, background, &retval ) )
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
FALCON_FUNC  process_systemCall( VMachine* vm )
{
   Item *command = vm->param(0);
   Item *mode = vm->param(1);

   if( command == 0 || ( !command->isString() &&  !command->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                             .extra("S|A{S}, [B]") );
   }


   bool background = mode == 0 ? false : mode->isTrue();
   GenericVector argv( &traits::t_stringptr_own() );

  if( command->isString() )
    Mod::argvize(argv, *command->asString());
  else
  {
     if ( !s_checkArray(command) )
        throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                              .extra( FAL_STR( proc_msg_allstr ) ) );
     s_appendCommands(argv, command);
  }
  argv.push( 0 );

   vm->idle();
   int retval;
   if( Sys::spawn( static_cast<String**>( argv.at(0) ),
                             false, background, &retval ) )
   {
      vm->unidle();
      vm->retval( retval );
   }
   else
   {
      vm->unidle();

      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
         .desc( FAL_STR( proc_msg_prccreate ) )
         .sysError( retval ) );
   }

}


/*#
   @function pread
   @brief Executes an external process and waits for its termination.
   @param command A string representing the program to be executed
        and its arguments, or an array whose first element is the program, and
        the others are the arguments.
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
FALCON_FUNC  process_pread( VMachine* vm )
{
   Item *command = vm->param(0);
   Item *mode = vm->param(1);

   if( command == 0 || (!command->isString() && !command->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                             .extra( "S|A{S}, B" ) );
   }

   bool background = mode == 0 ? false : mode->isTrue();
   GenericVector argv( &traits::t_stringptr_own() );

   argv.push( new String( Sys::shellName()) );
   argv.push( new String( Sys::shellParam()) );
   if( command->isString() )
      argv.push( new String( *command->asString() ) );
   else
   {
      if ( !s_checkArray(command) )
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                               .extra( FAL_STR( proc_msg_allstr ) ) );
      argv.push( new String( s_mergeCommandArray(command) ) );
   }
   argv.push( 0 );

   int retval = 0;
   CoreString* gs = new CoreString;
   if( Sys::spawn_read( static_cast<String**>( argv.at(0) ),
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
FALCON_FUNC  process_exec( VMachine* vm )
{
   Item *command = vm->param(0);

   if( command == 0 || ( !command->isString() &&  !command->isArray() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                            .extra("S|A{S}") );
   }

   GenericVector argv( &traits::t_stringptr_own() );
   if( command->isString() )
     Mod::argvize(argv, *command->asString());
   else
   {
      if ( !s_checkArray(command) )
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                               .extra( FAL_STR( proc_msg_allstr ) ) );
      s_appendCommands(argv, command);
   }
  argv.push( 0 );


   int retval;
   if( Sys::spawn( static_cast<String**>( argv.at(0) ),
                             true, false, &retval ) )
   {
      vm->retval( retval );
   }
   else
   {
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ ).
         desc( FAL_STR( proc_msg_prccreate ) ).sysError( retval ) );
   }
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

FALCON_FUNC  Process::init( VMachine* vm )
{
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );
   Item *command = vm->param(0);
   Item *mode_itm = vm->param(1);

   if( command == 0 || ( ! command->isString() && ! command->isArray() ) ||
      (mode_itm != 0 && ! mode_itm->isOrdinal())  )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
         .extra( "S|A{S}, [I]" ) );
   }

   // this will also work as flag, as it is valorized only when using the static args[] vector.
   uint32 mode = mode_itm == 0 ? 0 : (uint32) mode_itm->forceInteger();

   //pa_viaShell
   GenericVector argv( &traits::t_stringptr_own() );
   if ( (mode & 0x20) == 0x20 )
   {
     argv.push( new String( Sys::shellName()) );
     argv.push( new String( Sys::shellParam()) );

     if( ! command->isString()  )
       throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                             .extra( "S, [I]" ) );

     argv.push( new String( *command->asString() ) );
   }
   else
   {
     if( command->isString() )
       Mod::argvize(argv, *command->asString());
     else
     {
        if ( !s_checkArray(command) )
           throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
                                 .extra( FAL_STR( proc_msg_allstr ) ) );
        s_appendCommands(argv, command);
     }
   }
   argv.push( 0 );


   bool sinkin = ((mode & 0x1) == 0x1);
   bool sinkout = ((mode & 0x2) == 0x2);
   bool sinkerr = ((mode & 0x4) == 0x4);
   bool mergeerr = ((mode & 0x8) == 0x8);
   bool background = ((mode & 0x10) == 0x10);


   Sys::openProcess(self->handle(), static_cast<String**>( argv.at(0) ),
                    sinkin, sinkout, sinkerr, mergeerr, background );
   if ( self->handle()->lastError() != 0 )
   {
      throw new ProcessError( ErrorParam( FALPROC_ERR_CREATPROC, __LINE__ )
         .desc( FAL_STR(proc_msg_prccreate) )
         .sysError( self->handle()->lastError() ) );
   }
}

CoreObject* Process::factory(const CoreClass* cls, void* user_data, bool )
{
   return new Mod::Process(cls);
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

FALCON_FUNC  Process::wait( VMachine* vm )
{
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );
   vm->idle();
   if( ! self->handle()->wait( true ) )
   {
      vm->unidle();
      throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
         .desc( FAL_STR( proc_msg_waitfail ) )
         .sysError( self->handle()->lastError() ) );
   }
   else
   {
      self->handle()->close();
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
FALCON_FUNC  Process::terminate( VMachine* vm )
{
   Item *severe = vm->param(0);
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );

   if ( ! self->handle()->done() )
   {
      bool sev = severe == 0 ? false : severe->isTrue();
      if( ! self->handle()->terminate( sev ) )
      {
         throw new ProcessError( ErrorParam( FALPROC_ERR_TERM, __LINE__ ).
            desc( FAL_STR( proc_msg_termfail ) ).sysError( self->handle()->lastError() ) );
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
FALCON_FUNC  Process::value( VMachine* vm )
{
   Item *h_wait = vm->param(0);

   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );

   bool wait = h_wait == 0 ? false : h_wait->isTrue();
   if ( wait && ! self->handle()->done() )
   {
      vm->idle();
      if( ! self->handle()->wait( true ) )
      {
         self->handle()->close();
         vm->unidle();
         throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
            .desc( FAL_STR( proc_msg_waitfail ) )
            .sysError( self->handle()->lastError() ) );
      }
      else
         vm->unidle();
   }
   // give a test to see if the process is terminated in the meanwhile
   else if ( ! self->handle()->done() )
   {
      if( ! self->handle()->wait( false ) )
      {
         throw new ProcessError( ErrorParam( FALPROC_ERR_WAIT, __LINE__ )
            .desc( FAL_STR( proc_msg_waitfail ) )
            .sysError( self->handle()->lastError() ) );
      }
   }

   if( self->handle()->done() )
   {
      vm->retval( self->handle()->processValue() );
      self->handle()->close();
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
FALCON_FUNC  Process::getInput( VMachine* vm )
{
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );

   Stream *file = self->handle()->inputStream();
   if (file == 0 )
      vm->retnil();
   else
   {
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
FALCON_FUNC  Process::getOutput( VMachine* vm )
{
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );

   Stream *file = self->handle()->outputStream();
   if (file == 0 )
      vm->retnil();
   else
   {
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
FALCON_FUNC  Process::getAux( VMachine* vm )
{
   Mod::Process* self = Falcon::dyncast<Mod::Process*>( vm->self().asObject() );

   Stream *file = self->handle()->errorStream();
   if (file == 0 )
      vm->retnil();
   else
   {
      Item *stream_class = vm->findWKI( "Stream" );
      //if the rtl, that already returned File service, is right, this can't be zero.
      fassert( stream_class != 0 );
      CoreObject *co = stream_class->asClass()->createInstance();
      co->setUserData( file );
      vm->retval( co );
   }
}

void Process::registerExtensions( Module* self )
{
   Falcon::Symbol *proc_class = self->addClass( "Process", Process::init );
   proc_class->getClassDef()->factory( &Process::factory );
   self->addClassMethod( proc_class, "wait", Process::wait );
   self->addClassMethod( proc_class, "terminate", Process::terminate ).asSymbol()->
      addParam("severe");
   self->addClassMethod( proc_class, "value", Process::value ).asSymbol()->
      addParam("wait");
   self->addClassMethod( proc_class, "getInput", Process::getInput );
   self->addClassMethod( proc_class, "getOutput", Process::getOutput );
   self->addClassMethod( proc_class, "getAux", Process::getAux );
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

FALCON_FUNC  ProcessError::init( VMachine* vm )
{
   CoreObject *einst = vm->self().asObject();
   if( einst->getUserData() == 0 )
      einst->setUserData( new ProcessError );

   Falcon::core::Error_init( vm );
}

void ProcessError::registerExtensions( Module* self )
{
   Falcon::Symbol *error_class = self->addExternalRef( "Error" ); // it's external
   Falcon::Symbol *procerr_cls = self->addClass( "ProcessError", ProcessError::init );
   procerr_cls->setWKS( true );
   procerr_cls->getClassDef()->addInheritance(  new Falcon::InheritDef( error_class ) );
}

}} // ns Falcon::Ext

/* end of process_mod.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: process_ext.cpp

   Process module -- Falcon interface functions

   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab mar 11 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/process/process_fm.cpp"

/** \file
   Process module -- Falcon interface functions
   This is the module implementation file.
*/
#include <cstdio>
#include <falcon/vm.h>
#include <falcon/string.h>
#include <falcon/stream.h>
#include <falcon/itemarray.h>
#include <falcon/fassert.h>
#include <falcon/engine.h>
#include <falcon/stdhandlers.h>


#include "process_fm.h"
#include "process_mod.h"
#include "process.h"

#include <vector>

/*#
    @beginmodule process
*/

namespace Falcon {
namespace Feathers {

namespace {

   FALCON_DECLARE_FUNCTION( pid, "" );
   FALCON_DECLARE_FUNCTION( tid, "" );
   FALCON_DECLARE_FUNCTION( kill, "pid:N,severe:[B]" );
   FALCON_DECLARE_FUNCTION( system, "command:S,background:[B]" );
   FALCON_DECLARE_FUNCTION( systemCall, "command:S,background:[B],usePath:[B]" );
   FALCON_DECLARE_FUNCTION( pread, "command:S,background:[B],grabAux:[B]" );
   FALCON_DECLARE_FUNCTION( preadCall, "command:S,background:[B],grabAux:[B],usePath:[B]" );

/*#
   @funciton pid
   @brief Returns the process ID of the process hosting the Falcon VM.

   For command line Falcon interpreter, this ID may be considered the ID
   of the Falcon program being executed; in embedding applications, the
   function will return the process ID associated with the host application.
*/
FALCON_DEFINE_FUNCTION_P1(pid)
{
   ctx->returnFrame( Item().setInteger(Mod::processId()) );
}


/*#
   @funciton tid
   @brief Returns the thread ID of the process hosting the Falcon VM (if any).

   For command line Falcon interpreter, this ID may be considered the ID
   of the Falcon program being executed; in embedding applications, the
   function will return the thread ID associated with the thread running the current
   context in the host application.

   \note This number is not significant on many POSIX systems. A direct usage
   of a POSIX Pthread library wrapping may be necessary on those systems to
   have a consistent information.
*/
FALCON_DEFINE_FUNCTION_P1(tid)
{
   ctx->returnFrame( Item().setInteger(Mod::threadId()) );
}

/*#
   @function kill
   @brief Terminates the given process given its ID, if possible.
   @param pid The Process ID of the process that should be terminated.
   @optparam severe If given and true, use the maximum severity allowable to stop the given process.
   @return True on success, false on failure.

   The process having the given PID is terminated. On UNIX systems,
   a TERM signal is sent to the process. If severe is true, the process
   is stopped in the most hard way the system provides; i.e. in UNIX, KILL
   signal is sent.
*/

FALCON_DEFINE_FUNCTION_P1(kill)
{
   Item *id = ctx->param(0);
   Item *mode = ctx->param(1);

   if ( id == 0 || ! id->isOrdinal() )
   {
      throw paramError(__LINE__,SRC);
   }

   if ( mode == 0 || ! mode->isTrue() )
   {
      ctx->returnFrame( Item().setBoolean( Mod::processTerminate( id->forceInteger() ) ) );
   }
   else {
      ctx->returnFrame( Item().setBoolean( Mod::processKill( id->forceInteger() ) ) );
   }
}


static void internal_system( Function* func, VMContext* ctx, int mode, String* out = 0 )
{
   Item *i_command = ctx->param(0);

   if( i_command == 0 || (!i_command->isString() && !i_command->isArray() ) )
   {
      throw func->paramError(__LINE__, SRC);
   }
   ModuleProcess* mod = static_cast<ModuleProcess*>(func->fullModule());

   LocalRef<Mod::Process> prc( new Mod::Process(ctx, mod->classProcess() ) );
   const String& command = *i_command->asString();
   prc->open( command, mode, false );

   if( out != 0 )
   {
      Stream* streamOut = prc->outputStream();
      int count;
      char buffer[1024];
      while( (count = streamOut->read(buffer,1024) ) > 0 )
      {
         String temp;
         temp.adopt( buffer, count, 0 );
         out->append( temp );
      }

      prc->waitTermination();
      ctx->returnFrame( FALCON_GC_HANDLE(out) );
   }
   else {
      prc->waitTermination();
      ctx->returnFrame( (int64) prc->exitValue() );
   }

   prc->close();
}

/*#
   @function system
   @brief Executes an external process via command shell, and waits for its termination.
   @param command The command to be executed.
   @optparam background If given and true, the process runs hidden.
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

FALCON_DEFINE_FUNCTION_P1(system)
{
   int mode = Mod::Process::SINK_INPUT | Mod::Process::SINK_OUTPUT | Mod::Process::SINK_AUX
               | Mod::Process::USE_SHELL;

   Item *i_mode = ctx->param(1);
   if( i_mode != 0 && i_mode->isTrue() )
   {
      mode |= Mod::Process::BACKGROUND;
   }

   internal_system( this, ctx, mode );
}

/*#
   @function systemCall
   @brief Executes an external process and waits for its termination.
   @param command The command to be executed.
   @optparam background If given and true, the process runs hidden.
   @optparam usePath If given and true, use the system path to search for the given process.
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
FALCON_DEFINE_FUNCTION_P1(systemCall)
{
   int mode = Mod::Process::SINK_INPUT | Mod::Process::SINK_OUTPUT | Mod::Process::SINK_AUX;

   Item *i_mode = ctx->param(1);
   Item *i_usePath = ctx->param(2);
   if( i_mode != 0 && i_mode->isTrue() )
   {
      mode |= Mod::Process::BACKGROUND;
   }

   if( i_usePath != 0 && i_usePath->isTrue() )
   {
      mode |= Mod::Process::USE_PATH;
   }

   internal_system( this, ctx, mode );
}

static void internal_pread( Function* func, VMContext* ctx, bool useShell  )
{
   Item *i_mode = ctx->param(1);
   Item *i_aux = ctx->param(2);
   Item *i_usePath = ctx->param(3);

   int mode = Mod::Process::SINK_INPUT;
   if( useShell )
   {
      mode |= Mod::Process::USE_SHELL;
   }
   else if( i_usePath != 0 && i_usePath->isTrue() )
   {
      mode |= Mod::Process::USE_PATH;
   }

   if( i_aux != 0 && i_aux->isTrue() )
   {
      mode |= Mod::Process::MERGE_AUX;
   }
   else {
      mode |= Mod::Process::SINK_AUX;
   }

   if( i_mode != 0 && i_mode->isTrue() )
   {
      mode |= Mod::Process::BACKGROUND;
   }

   String* result = new String();
   try {
      internal_system( func, ctx, mode, result );
   }
   catch( ... )
   {
      delete result;
      throw;
   }
}

/*#
   @function pread
   @brief Executes an external process and waits for its termination.
   @param command A string representing the program to be executed and its arguments.
   @optparam background If given and true, the process runs hidden.
   @optparam grabAux If given and true, read also the stderr stream of the process.

   @return The full output generated by the process.
   @raise ProcessError if the process couldn't be created.

   This function launches an external system command and waits until the command
   execution is terminated, returning the exit code of the child process.

   for example:

   @code
   dir_contents = pread( "ls", true, true )
   > dir_contents
   @endcode

   If the process cannot be started if it fails to start, an error is raised.

   \note This function uses the standard system shell to execute the passed
   commands, so it is possible to pipe applications and redirect streams
   via the standard "|" and ">" command line characters.

   @note This function uses the available command shell. To invoke the
   command directly, use the preadCall function.
*/
FALCON_DEFINE_FUNCTION_P1(pread)
{
   internal_pread( this, ctx, true );
}


/*#
   @function preadCall
   @brief Executes an external process and waits for its termination.
   @param command A string representing the program to be executed and its arguments.
   @optparam background If given and true, the process runs hidden.
   @optparam grabAux If given and true, read also the stderr stream of the process.
   @optparam usePath If given and true, use the system path to search for the given process.

   @return The full output generated by the process.
   @raise ProcessError if the process couldn't be created.

   This function is equivalent to @a pread, but it invokes directly the given
   command without using the system shell to parse the command line.
*/

FALCON_DEFINE_FUNCTION_P1(preadCall)
{
   internal_pread( this, ctx, false );
}

/*#
   @class Process
   @brief Execute and control child processes.

   This class is meant for finer control of child processes and
   inter-process communication.
*/

/*#
 @method open Process

   The process named in the @b command argument is started. It is possible to
   provide either a string containing a complete command line, with the process
   name and its arguments, or an array whose first element is the process name,
   and the other elements are the parameters that will be provided to the
   process.

   The optional @b flags parameter can control the behavior of the started process,
   and may be a combination of the followings:

   - SINK_INPUT: prevent the child process to wait for input from us.
   - SINK_OUTPUT: destroy all the child process output.
   - SINK_AUX: destroy all the child process auxiliary stream output.
   - MERGE_AUX: merge output and auxiliary stream so that they are read
                        by just reading the output stream. If given, SINK_AUX
                        is ignored.
   - BACKGROUND: Put the process in background/hidden mode.
   - USE_SHELL: Use host system shell to launch the process or execute the command.
         If given, USE_PATH is ignored.
   - USE_PATH: Try to find the given command relative to the
               current system search path.

*/

}

namespace CProcess {

FALCON_DECLARE_FUNCTION( open, "command:S,mode:[N]" )
FALCON_DEFINE_FUNCTION_P1( open )
{
   Item *i_command = ctx->param(0);
   Item *i_mode = ctx->param(1);

   if( i_command == 0 || ! i_command->isString()
      || (i_mode != 0 && ! i_mode->isOrdinal())  )
   {
      throw paramError(__LINE__,SRC);
   }

   Mod::Process* prc = static_cast<Mod::Process*>(ctx->self().asInst());
   const String& command = *i_command->asString();
   int mode = i_mode != 0 ? (int32) i_mode->forceInteger() : 0;
   prc->open( command, mode, true );
   ctx->returnFrame();
}


/*#
   @method wait Process
   @brief Waits for a child process to terminate.
   @raise ProcessError on system errors or wait failed.

   Waits for the child process to terminate cleanly.

   @note It's preferable to use a waiter to wait on the process termination.
*/


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
FALCON_DECLARE_FUNCTION( terminate, "severe:[B]" )
FALCON_DEFINE_FUNCTION_P1( terminate )
{
   Item *i_severe = ctx->param(0);
   Mod::Process* self = static_cast<Mod::Process*>( ctx->self().asInst() );

   bool severe = i_severe != 0 && i_severe->isTrue();
   self->terminate( severe );
}


/*#
   @property exitValue Process
   @brief Retrieves exit value of the child process.

   If the child process has completed its execution, this value will be the number
   that was yielded by the process at its termination.

   If the process is terminated by a an external signal or an internal error,
   the value will less than 0.

   This value will be @b nil if the process has not yet terminated.

   @note Usually, the value 0 is used to indicate a correct completion.
*/
static void get_exitValue( const Class*, const String&, void* instance, Item& value )
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   int ev = 0;
   if( self->exitValue(ev) )
   {
      value.setInteger(ev);
   }
   else {
      value.setNil();
   }
}

/*#
 @property pid Process
 @brief the PID of the child process (if started).

 This number will be 0 if the method open is not yet called,
 or a child process ID after the method is called.

 */
static void get_pid( const Class*, const String&, void* instance, Item& value )
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   value.setInteger( self->pid() );
}


/*#
   @property input Process
   @brief The process input stream.

   The returned stream can be used as a Falcon stream,
   but it supports only write operations.

   If the process has been opened with the SINK_IN flag,
   this property is @b nil.
*/
static void get_input( const Class*, const String&, void* instance, Item& value )
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   Stream* stream = self->inputStream();
   if( stream != 0 )
   {
      stream->incref();
      value.setUser( FALCON_GC_HANDLE(stream) );
   }
   else {
      value.setNil();
   }
}

/*#
   @property output Process
   @brief The process output stream.

   The returned stream can be used as a Falcon stream,
   but it supports only read operations.

   If the process has been opened with the SINK_OUTPUT flag,
   property is @b nil.
*/
static void get_output( const Class*, const String&, void* instance, Item& value )
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   Stream* stream = self->outputStream();
   if( stream != 0 )
   {
      stream->incref();
      value.setUser( FALCON_GC_HANDLE(stream) );
   }
   else {
      value.setNil();
   }
}

/*#
   @property aux Process
   @brief The process error/auxiliary stream.

   The returned stream can be used as a Falcon stream,
   but it supports only read operations.

   If the process has been opened with the SINK_AUX flag,
   property is @b nil.
*/
static void get_aux( const Class*, const String&, void* instance, Item& value )
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   Stream* stream = self->outputStream();
   if( stream != 0 )
   {
      stream->incref();
      value.setUser( FALCON_GC_HANDLE(stream) );
   }
   else {
      value.setNil();
   }
}
}


ClassProcess::ClassProcess():
         ClassShared("Process")
{
   setParent(Engine::instance()->stdHandlers()->sharedClass());

   addProperty( "input", &CProcess::get_input );
   addProperty( "output", &CProcess::get_output );
   addProperty( "aux", &CProcess::get_aux );
   addProperty( "exitValue", &CProcess::get_exitValue );
   addProperty( "pid", &CProcess::get_pid );

   addMethod( new CProcess::Function_open );
   addMethod( new CProcess::Function_terminate );

   addConstant( "SINK_INPUT", (Falcon::int64) Mod::Process::SINK_INPUT );
   addConstant( "SINK_OUTPUT", (Falcon::int64) Mod::Process::SINK_OUTPUT );
   addConstant( "SINK_AUX", (Falcon::int64) Mod::Process::SINK_AUX );
   addConstant( "MERGE_AUX", (Falcon::int64) Mod::Process::MERGE_AUX );
   addConstant( "BACKGROUND", (Falcon::int64) Mod::Process::BACKGROUND );
   addConstant( "USE_SHELL", (Falcon::int64) Mod::Process::USE_SHELL );
   addConstant( "USE_PATH", (Falcon::int64) Mod::Process::USE_PATH );
}

ClassProcess::~ClassProcess()
{
}

void* ClassProcess::createInstance() const
{
   return FALCON_CLASS_CREATE_AT_INIT;
}


bool ClassProcess::op_init( VMContext* ctx, void* instance, int32 pcount ) const
{
   instance = new ::Falcon::Mod::Process(ctx, this);
   ctx->stackResult(pcount+1, Item(FALCON_GC_STORE(this, instance )) );
   return true;
}

void* ClassProcess::clone( void* instance ) const
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   self->incref();
   return self;
}

void ClassProcess::dispose( void* instance ) const
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   self->decref();
}

void ClassProcess::describe( void* instance, String& target, int, int maxlen ) const
{
   Mod::Process* self = static_cast<Mod::Process*>( instance );
   target += "Process \"" + self->cmd() + "\" ";
   int ev = 0;
   if( self->exitValue( ev ) )
   {
      target.A(" - terminated ");
      target.N( ev );
   }

   if( maxlen > 4 && (target.length() + 4 > (unsigned)maxlen) )
   {
      target = target.subString( maxlen - 4 ) + " ...";
   }
}



//=====================================================================
// Process enumeration
//=====================================================================

namespace CProcessEnum {
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
      > enum.name, " - ", enum.cmdLine, ": ", enum.pid, " ( child of ", enum.ppid, ")"
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

FALCON_DECLARE_FUNCTION( next, "" )
FALCON_DEFINE_FUNCTION_P1( next )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( ctx->self().asInst() );
   int64 res = (int64) self->next();

   if ( res != 1 )
   {
      if ( res == -1 )
      {
         throw FALCON_SIGN_XERROR( ProcessError, FALCON_PROCESS_ERROR_ERRLIST,
            .desc( FALCON_PROCESS_ERROR_ERRLIST_MSG ) );
      }
   }

   ctx->returnFrame(res);
}


/*#
   @method close ProcessEnum
   @brief Closes the enumeration freeing system resources.

   Disposes the data associated with this item without waiting
   for the garbage collector to reclaim them.
*/
FALCON_DECLARE_FUNCTION( close, "" )
FALCON_DEFINE_FUNCTION_P1( close )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( ctx->self().asInst() );
   self->close();
}

static void get_pid( const Class*, const String&, void* instance, Item& value )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( instance );
   value.setInteger( (int64) self->pid() );
}

static void get_ppid( const Class*, const String&, void* instance, Item& value )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( instance );
   value.setInteger( (int64) self->ppid() );
}

static void get_cmdLine( const Class*, const String&, void* instance, Item& value )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( instance );
   value = FALCON_GC_HANDLE( new String(self->cmdLine()) );
}

static void get_name( const Class*, const String&, void* instance, Item& value )
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>( instance );
   value = FALCON_GC_HANDLE( new String(self->name()) );
}

}


ClassProcessEnum::ClassProcessEnum():
         Class("ProcessEnum")
{
   setParent(Engine::instance()->stdHandlers()->sharedClass());

   addProperty( "name", &CProcessEnum::get_name );
   addProperty( "cmdLine", &CProcessEnum::get_cmdLine );
   addProperty( "pid", &CProcessEnum::get_pid );
   addProperty( "ppid", &CProcessEnum::get_ppid );

   addMethod( new CProcessEnum::Function_next );
   addMethod( new CProcessEnum::Function_close );

}

ClassProcessEnum::~ClassProcessEnum()
{
}

void* ClassProcessEnum::createInstance() const
{
   return new Mod::ProcessEnum;
}


bool ClassProcessEnum::op_init( VMContext* , void* , int32  ) const
{
   return false;
}

void* ClassProcessEnum::clone( void* ) const
{
   return 0;
}

void ClassProcessEnum::dispose( void* instance ) const
{
   Mod::ProcessEnum* self = static_cast<Mod::ProcessEnum*>(instance );
   delete self;
}

void ClassProcessEnum::describe( void*, String& target, int, int ) const
{
   target = "ProcessEnum";
}

//=================================================================
// Process error
//=================================================================

/*#
   @class ProcessError
   @brief Error generated by process related system failures.
   @optparam code A numeric error code.
   @optparam description A textual description of the error code.
   @optparam extra A descriptive message explaining the error conditions.
   @from Error code, description, extra

   See the Error class in the core module.
*/

//=================================================================
// Process module
//=================================================================

ModuleProcess::ModuleProcess():
         Module(FALCON_FEATHER_PROCESS_NAME)
{
   m_classProcess = new ClassProcess;

   *this
         << new Function_kill
         << new Function_pid
         << new Function_tid
         << new Function_system
         << new Function_systemCall
         << new Function_pread
         << new Function_preadCall
         << m_classProcess
         << new ClassProcessError
         << new ClassProcessEnum
      ;
}

ModuleProcess::~ModuleProcess()
{
}

}} // ns Falcon::Ext

/* end of process_mod.cpp */

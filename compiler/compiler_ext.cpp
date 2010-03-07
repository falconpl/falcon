/*
   FALCON - The Falcon Programming Language
   FILE: compiler_ext.cpp

   Compiler module main file - extension implementation.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler module main file - extension implementation.
*/

#include <falcon/setup.h>
#include <falcon/module.h>
#include <falcon/vm.h>
#include <falcon/rosstream.h>
#include <falcon/attribmap.h>
#include <falcon/lineardict.h>
#include <falcon/pcode.h>

#include "compiler_ext.h"
#include "compiler_mod.h"
#include "compiler_st.h"

/*#
   @beginmodule feather_compiler
*/

namespace Falcon {

namespace Ext {

/*#
   @class _BaseCompiler
   @brief Abstract base class for the @a Compiler and @a ICompiler classes.

   This base class is used to put some common properties at disposal of both
   the subclasses: the standard @a Compiler, and the @a ICompiler (Incremental Compiler)
   that allows live evaluation in a protected children virtual machine.

   @prop alwaysRecomp If true, a load method finding a valid .fam that may
       substitute a .fal will ignore it, and will try to compile and
       load the .fal instead.

   @prop compileInMemory If true (the default) intermediate compilation
      steps are performed in memory. If false, temporary files are used instead.

   @prop ignoreSources If true, sources are ignored, and only .fam or
      shared object/dynamic link libraries will be loaded.

   @prop path The search path for modules loaded by name. It's a set of
      Falcon format paths (forward slashes to separate dirs, e.g. "C:/my/path"),
      separated by semi comma.

   @prop saveMandatory If true, when saveModule option is true too and a
      module can't be serialized, the compiler raises an exception.

   @prop saveModules If true, once compiled a source that is located on a
      local file system, the compiler will also try to save the .fam pre-compiled
      module, that may be used if the same module is loaded a
      second time. Failure in saving the pre-compiled module is not reported,
      unless saveMandatory option is set.

   @prop sourceEncoding The encoding of the source file. It defaults to
      default system encoding that Falcon is able to detect. Use one of the
      encoding names known by the Transcoder class.

   @prop launchAtLink If true, the __main__ function (that is, the entry point)
         of the loaded modules is executed before returning it. This allows
         the modules to initalize themselves and set their global variables.
         Notice that this step may be autonomusly performed also by the loader
         after the loading is complete.

   @prop language Language code used to load language-specific string tables.
      When this entry is valorized to a valid international language code, as i.e.
      "en_US", the compiler tries to use .ftr files found besides their modules to
      alter their string tables, changing the original strings with their
      translation for the desired language. If the language table file or the
      required translation is not available, the operation silently fails and
      the module is loaded with the string untranslated.
*/

/*#
   @method addFalconPath _BaseCompiler
   @brief Adds the default system paths to the path searched by this compiler.

   This method instructs the compiler that the default search path used by
   Falcon engine should be also searched when loading modules. This means
   that the directory in which official Falcon modules are stored, or
   those set in the FALCON_LOAD_PATH environment variables, or compiled
   in for a particular installation of Falcon, will be searched whenever
   loading a module.

   The paths are inserted at the beginning; so, they will be the first
   searched. It is possible then to alter the search path by changing
   the @a Compiler.path property and i.e. prepending a desired local
   search path to it.
*/

/*#
   @method setDirective _BaseCompiler
   @brief Compiles a script on the fly.
   @param dt Directive to be set.
   @param value Value to be given to the directive.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical error.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Sets a directive as if the scripts that will be loaded by this compiler defined it
   through the directive statement. Scripts can always override a directive by setting
   it to a different value in their code; notice also that compilation directives are
   useful only if a compilation actually take places. In case a .fam or a binary module
   is loaded, they have no effect.
*/

FALCON_FUNC BaseCompiler_setDirective( ::Falcon::VMachine *vm )
{
   Item *i_directive = vm->param( 0 );
   Item *i_value = vm->param( 1 );

   if( i_directive == 0 || ! i_directive->isString() ||
       i_value == 0 || ( ! i_value->isString() && ! i_value->isOrdinal() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,S|N" ) );
   }

   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );

   if ( i_value->isString() )
      iface->loader().compiler().setDirective( *i_directive->asString(), *i_value->asString() );
   else
      iface->loader().compiler().setDirective( *i_directive->asString(), i_value->forceInteger() );

   // in case of problems, an error is already raised.
}
FALCON_FUNC BaseCompiler_addFalconPath( ::Falcon::VMachine *vm )
{
   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );
   iface->loader().addFalconPath();

}


/*#
   @class Compiler
   @from _BaseCompiler
   @brief Main interface to the reflexive compiler.
   @optparam path The default search path of the compiler.

   The static compiler class is an interface to the compilation facilities
   and the vritual machine currently running the caller Falcon program.

   Compiled resources (either external binary modules or falcon scripts, that
   can be stored on external resources or compiled on-the-fly) are integrated
   in the running virtual machine as a separate module, immediately linked and
   made runnable. The caller script receives a @a Module instance which can be
   used to control the execution of the target module, or to unload it at a
   later time.

   The linked modules receive every globally exported symbol that is accessible
   to the caller script, but their export requests are ignored (they can't modify
   the global execution environment of the calling script).

   Although a single compiler should be enough for the needs of a simple script
   program, it is possible to create as many instances of the compiler as needed.

   However, it is sensible to instance this class through singleton objects,
   so that they get prepared by the link step of the VM:

   @code
      load compiler

      object MyCompiler from Compiler
      end
   @endcode

   @note If @b path is not provided, defaults to "."
   (script current working directory).
*/

FALCON_FUNC Compiler_init( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param( 0 );

   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );
   if( i_path != 0 )
   {
      if( ! i_path->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) );
         return;
      }

      iface->loader().setSearchPath( *i_path->asString() );
   }
   else
      iface->loader().setSearchPath( Engine::getSearchPath() );
}



void internal_link( ::Falcon::VMachine *vm, Module *mod, CompilerIface *iface )
{

   Runtime rt( &iface->loader(), vm );
   rt.hasMainModule(false);
   // let's try to link
   rt.addModule( mod, true );

   bool ll = vm->launchAtLink();
   LiveModule* lmod = 0;

   // avoid a re-throw in the fast-path
   if ( iface->launchAtLink() != ll )
   {
      vm->launchAtLink( iface->launchAtLink() );

      try {
         lmod = vm->link( &rt );
         vm->launchAtLink( ll );
      }
      catch( ... )
      {
         vm->launchAtLink( ll );
         throw;
      }
   }
   else
      lmod = vm->link( &rt );

   // ok, the module is up and running.
   // wrap it
   Item *mod_class = vm->findWKI( "Module" );
   fassert( mod_class != 0 );
   CoreObject *co = mod_class->asClass()->createInstance();
   // we know the module IS in the VM.
   co->setUserData( new ModuleCarrier( lmod ) );

   co->setProperty( "name", mod->name() );
   co->setProperty( "path", mod->path() );

   // return the object
   vm->retval( co );

   // we can remove our reference
   mod->decref();
}

/*#
   @method compile Compiler
   @brief Compiles a script on the fly.
   @param modName A logical unique that will be given to the module after compilation.
   @param data The data to compile. It may be a string or a stream valid for input.
   @return On success, a @a Module instance that contains the compiled module.
   @raise SyntaxError if the module contains logical error.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Tries to compile the module in the @b data parameter. On failure, a SyntaxError
   is raised; the subErrors member of the returned error will contain an array
   where every single compilation error is specified.

   On success, an instance of Module class is returned.
*/
FALCON_FUNC Compiler_compile( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );
   // The parameter may be  a string or a stream
   Item *i_data = vm->param( 1 );

   if( i_name == 0 || ! i_name->isString() ||
      i_data == 0 || (! i_data->isString() && ! i_data->isObject()) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S|Stream" ) );
   }

   Stream *input;
   String *name = i_name->asString();
   bool bDelete;

   // now, if data is an object it must be a stream.
   if( i_data->isObject() )
   {
      CoreObject *data = i_data->asObject();
      if ( ! data->derivedFrom( "Stream" ) )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "S, S|Stream" ) );
      }

      // ok, get the stream
      input = (Stream *) data->getUserData();
      bDelete = false;
   }
   else {
      // if it's a string, we have to create a stream
      input = new ROStringStream( *i_data->asString() );
      bDelete = true;
   }

   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );

   Module *mod = 0;

   bool bSave = iface->loader().saveModules();
   try
   {
      iface->loader().saveModules( false );
      mod = iface->loader().loadSource( input, *name, *name );
      iface->loader().saveModules( bSave );
      internal_link( vm, mod, iface );
      // don't decref, on success internal_link does.
   }
   catch(Error* err)
   {
      iface->loader().saveModules( bSave );
      CodeError *ce = new CodeError( ErrorParam( e_loaderror, __LINE__ ).
         extra( *i_name->asString() ) );

      ce->appendSubError(err);
      err->decref();

      if ( mod != 0 )
         mod->decref();
      throw ce;
   }

   if( bDelete )
      delete input;
}

/*#
   @method loadByName Compiler
   @brief Loads a module given its logical name.
   @param modName The logical name of the module to be loaded.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical error.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Tries to load a logically named module scanning for suitable sources, pre-compiled
   modules and binary modules in the search path. In case a suitable module
   cannot be found, the method returns nil. If a module is found, a
   CodeError is raised in case compilation or link steps fails.
*/
FALCON_FUNC Compiler_loadByName( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );

   const Symbol *caller_sym;
   const Module *caller_mod;
   String modname;
   if ( vm->getCaller( caller_sym, caller_mod ) )
      modname = caller_mod->name();

   Module *mod = 0;
   try
   {
      mod = iface->loader().loadName( *i_name->asString(), modname );
      internal_link( vm, mod, iface );
      // don't decref, on success internal_link does.
   }
   catch(Error* err)
   {
      CodeError *ce = new CodeError( ErrorParam( e_loaderror, __LINE__ ).
         extra( *i_name->asString() ) );

      ce->appendSubError(err);
      err->decref();

      if ( mod != 0 )
         mod->decref();
      throw ce;
   }
}

/*#
   @method loadFile Compiler
   @brief Loads a Falcon resource from a location on the filesystem.
   @param modPath Relative or absolute path to a loadable Falcon module or source.
   @optparam alias Alias under which the module should be loaded.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical error.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Loads the given file, trying to perform compilation or loading of the relevant
   .fam precompiled module depending on the property settings. In example, if
   loading "./test.fal", unless alwaysRecomp property is true, "./test.fam" will be
   searched too, and if it's found and newer than ./test.fal, it will be loaded
   instead, skipping compilation step. Similarly, if "./test.fam" is searched,
   unless ignoreSource is true, "./test.fal" will be searched too, and if it's newer
   than ./test.fam it will be recompiled.

   If @b alias parameter is given, the loaded modules assumes the given name.
   The same naming conventions used by the load directive (names starting
   with a single "." or with "self.") are provided. Notice that the path
   separators are NOT automatically transformed into "." in the module
   logical name, so to import the module under a local namespace, using
   this parameter is essential.

   In case a suitable module cannot be found, the method returns nil. If a module is found,
   a CodeError is raised in case compilation or link steps fails.
*/
FALCON_FUNC Compiler_loadFile( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );
   Item *i_alias = vm->param( 1 );

   if( i_name == 0 || ! i_name->isString()
       || ( i_alias != 0 && !i_alias->isString() ) )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ )
            .extra( "S,[S]" ) );
   }

   CompilerIface *iface = dyncast<CompilerIface*>( vm->self().asObject() );
   Module *mod = 0;

   try {
      mod = iface->loader().loadFile( *i_name->asString() );

      // select the module name -- if no alias is given get the official name
      const Symbol *caller_sym;
      const Module *caller_mod;
      String parent_name;
      if ( vm->getCaller( caller_sym, caller_mod ) )
         parent_name = caller_mod->name();
      String nmodName = Module::absoluteName(
               i_alias == 0 ? mod->name() : *i_alias->asString(),
               parent_name );
      mod->name( nmodName );

      internal_link( vm, mod, iface );
      // don't decref, on success internal_link does.
   }
   catch(Error* err)
   {
      CodeError *ce = new CodeError( ErrorParam( e_loaderror, __LINE__ ).
         extra( *i_name->asString() ) );

      ce->appendSubError(err);
      err->decref();

      if ( mod != 0 )
         mod->decref();
      throw ce;
   }
}

//=========================================================
// Incremental compiler
//

/*#
   @class ICompiler
   @from _BaseCompiler
   @brief Interaface to incremental evaluator.
   @optparam path The default search path of the compiler.

   The incremental compiler, or "evaluator", is meant to provide a falcon-in-falcon
   compilation environment.

   @note The incremental compiler is currently under development, and subject to
         sudden changes in behaviors and interfaces.

   While the @a Compiler class is meant to help scripts to load and use foreign code
   in their context, the @b ICompiler class provides it's own private virtual machine
   and executes all the code it receives in a totally separate environment.

   Compiling a script through the @b ICompiler is phisically equivalent to start a new
   'falcon' command line interpreter and ordering it to run a script, with two main
   differences: first, the ICompiler instance runs serially with the caller in the
   same process, and second, the ICompiler also allows incremental compilation.

   Incremental compilation means that it's possible to evaluate falcon statements
   incrementally as they are compiled on the fly and executed one by one.

   Compilation methods @a ICompiler.compileNext and @a ICompiler.compileAll returns a
   compilation state (or eventually raise an error), that is useful to determine
   what happened and what action to take after a partial compilation. Possible
   return values are:

   - @b ICompiler.NOTHING - No operation performed (i.e. returned for just comments or whitespaces).
   - @b ICompiler.MORE - The statement is not complete and requires more input.
   - @b ICompiler.INCOMPLETE - While the last statement is complete, the context is still open and requires
                      some more "end" to be closed.
   - @b ICompiler.DECL - The compiler parsed and commited a complete declaration (top level function,
                  class, object etc).
   - @b ICompiler.STATEMENT - A toplevel statement was parsed and executed. Loops, branches, and non-expression
           statements fall in this category.
   - @b ICompiler.EXPRESSION - A single complete expression was parsed. The evaluated result is available through
                      the @a ICompiler.result property.
   - @b ICompiler.CALL - It was determined that the expression was a single call, in the form <exp1>(<exp2>).
         Some may want to know this information to avoid printing obvious results (calls returning nil
         are porbably better to be handled silently).
   - @b ICompiler.TERMINATED - The virtual machine has been requested to terminate.

    When the functions return MORE or INCOMPLETE, no operation is actually performed. The caller should
    provide new input with more data adding it to the previously parsed one, like in the following example:

    @code
    load compiler

    ic = ICompiler()
    str = "printl( 'hello',"  // incomplete
    > ic.compileNext( str )   // will do nothing and return 2 == ICompiler.MORE
    str += " 'world')\n"      // add \n for a complete statement
    > ic.compileNext( str )   // will do the print return 6 == ICompiler.CALL
    @endcode

    Everything happening in @a ICompiler.compileNext and @a ICompiler.compileAll happens in a
    separate virtual machine which is totally unrelated with the calling one. Nevertheless,
    they can safely share the values in the @a ICompiler.result property:

    @code
    load compiler

    ic = ICompiler()
    ic.compileNext( "a = [1,2,3,4]\n" )
    > inspect( ic.result )   // will be the array created by the statement
    ic.result[0] = "Changed"
    ic.compileNext( "inspect( a )\n" )   // will show the change in 'a'
    @endcode

    @note Always remember to add a \n or ';' after a complete statement in incremental
          compilation (this requirement might be removed in future).

    @note Don't try to share and call callable symbols. Chances are that they may be
          callable in one VM, but unaccessible from the other.

    Setting the streams in @a ICompiler.stdIn, @a ICompiler.stdOut and @a ICompiler.stdErr
    to new values, it is possible to intercept output coming from the virtual machine used
    by the incremental compiler, and to feed different input into it. This is the mechanism
    used by the macro compiler.

    Accessing the stream stored one of the @b ICompiler properties, an usable base class
    @a Stream clone will be returned; this may differ from the object that was originally stored
    in the property. For example, suppose you want to capture all the output generated by the
    scripts you incrementally compile through a @a StringStream. Once set, accessing the @b stdOut
    property will return a standard stream, which is actually a shell pointing to your same
    StringStream. To use StringStream specific methods, as i.e. @a StringStream.closeToString,
    you need to have a reference to the original object, otherwise you'll need to use the
    standard Stream methods to get the data (i.e. seek(0) and grabText()).

    @code
    load compiler

    ic = ICompiler()
    ss = StringStream()    // to keep our string stream

    // perform redirection
    ic.stdOut = ss
    ic.compileNext( "> 'Hello world';" )

    // get the result, but through our original item.
    > ss.getString()       // prints Hello world
    @endcode

    @prop stdIn standard input stream of the incremental compiler virtual machine.
    @prop stdOut standard output stream of the incremental compiler virtual machine.
    @prop stdErr standard error stream of the incremental compiler virtual machine. This is
          used by default error reporting and informative functions as inspect().
    @prop result Item containing last evaluation result.
*/


FALCON_FUNC ICompiler_init( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param( 0 );

   ICompilerIface *iface = dyncast<ICompilerIface*>( vm->self().asObject() );
   if( i_path != 0 )
   {
      if( ! i_path->isString() )
      {
         throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) );
      }

      iface->loader().setSearchPath( *i_path->asString() );
   }
   else
      iface->loader().setSearchPath( Engine::getSearchPath() );
}


/*#
   @method compileNext ICompiler
   @brief Compiles and executes at most one complete statement or declaration.
   @param code A string or a @a Stream containing at least a complete line of code.
   @return One of the enumeration values in @a ICompiler return values.
   @raise CodeError on compilation error.
   @raise Error (any kind of error) on runtime error.

   This method reads exactly one statement (up to the next \n or ';') and executes it
   immediately.

   One or more compilation errors will cause a CodeError containing all the detected
   errors to be raised.

   A runtime error will be re-thrown in the context of the calling program.

   The method returns a number representing the kind of code detected and eventually
   executed by the interactive compiler. For more details, see the description of
   the @a ICompiler class.
*/

FALCON_FUNC ICompiler_compileNext( ::Falcon::VMachine *vm )
{
   Item *i_code = vm->param( 0 );

   ICompilerIface *iface = dyncast<ICompilerIface*>( vm->self().asObject() );
   if( i_code != 0 )
   {
      if( i_code->isString() )
      {
         InteractiveCompiler::t_ret_type ret = iface->intcomp()->compileNext( *i_code->asString() );
         vm->retval( (int64) ret );
         return;
      }
      else if ( i_code->isObject() && i_code->asObjectSafe()->derivedFrom( "Stream" ) )
      {
         InteractiveCompiler::t_ret_type ret = iface->intcomp()->compileNext(
            dyncast<Stream*>(i_code->asObject()->getFalconData()) );
         vm->retval( (int64) ret );
         return;
      }
   }

   throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S|Stream" ) );
}

/*#
   @method compileAll ICompiler
   @brief Compiles entierely the given input.
   @param code A string containing a complete program (even small).
   @return One of the enumeration values in @a ICompiler return values.
   @raise CodeError on compilation error.
   @raise Error (any kind of error) on runtime error.

   This method reads exactly as many statements as possible, compiles them and runs
   them on the fly.

   One or more compilation errors will cause a CodeError containing all the detected
   errors to be raised.

   A runtime error will be re-thrown in the context of the calling program.

   The method returns a number representing the kind of code detected and eventually
   executed by the interactive compiler. For more details, see the description of
   the @a ICompiler class.
*/
FALCON_FUNC ICompiler_compileAll( ::Falcon::VMachine *vm )
{
   Item *i_code = vm->param( 0 );

   ICompilerIface *iface = dyncast<ICompilerIface*>( vm->self().asObject() );
   if( i_code != 0 )
   {
      if( i_code->isString() )
      {
         InteractiveCompiler::t_ret_type ret = iface->intcomp()->compileAll( *i_code->asString() );
         vm->retval( (int64) ret );
         return;
      }
   }

   throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
}

/*
   @method reset ICompiler
   @brief Resets the compiler.

   This method destroys all the entities declared by the module and clears the
   references held by its virtual machine. It's practically equivalent to create
   a new instance of the ICompiler, with less overhead.
*/

FALCON_FUNC ICompiler_reset( ::Falcon::VMachine *vm )
{
   ICompilerIface *iface = dyncast<ICompilerIface*>( vm->self().asObject() );
   iface->intcomp()->reset();
}

//=========================================================
// Module
//

/*#
   @class Module
   @brief Handle to loaded modules.

   The module class is a handle by which the scripts can read or write
   other modules symbols. It should not be instantiated directly by scripts;
   module class instances are returned by the @a Compiler methods.
*/

/*#
   @method get Module
   @brief Retreives a value from the target module.
   @param symName The name of the symbol to be loaded.
   @raise AccessError if the symbol name is not found.

   If the module provides a global symbol with the given name,
   the value of the symbol is returned. Nil may be a valid return
   value; in case the symbol is not found, an AccessError is raised.
*/

FALCON_FUNC Module_get( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      throw new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) );
   }

   vm->retval( *itm );
}

/*#
   @method set Module
   @brief Changes a value in the target module.
   @param symName The name of the symbol to be changed.
   @param value The new value to be set.
   @raise AccessError if the symbol name is not found.

   Sets the value of a global symbol in the target module. The value may
   be any kind of Falcon item, including a reference, in which case any
   change to the local value is immediately reflected in the target module.

   In case the symbol is not found, an AccessError is raised.
*/
FALCON_FUNC Module_set( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );
   Item *i_value = vm->param( 1 );

   if( i_name == 0 || ! i_name->isString() || i_value == 0 )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,X" ) );
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0|| ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      throw new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) );
   }

   *itm = *i_value;
}

/*#
   @method getReference Module
   @brief Retreives a value from the target module.
   @param symName The name of the symbol to be referenced.
   @return A reference to the desired item.
   @raise AccessError if the symbol name is not found.

   Loads a reference to the given symbol in the target module. On success,
   after this call assignments to the returned variable will be immediately
   reflected in both the modules, be them applied in the caller or in the
   loaded module. To break the reference, use the $0 operator as usual.

   In case the symbol is not found, an AccessError is raised.
*/
FALCON_FUNC Module_getReference( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      throw new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) );
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      throw new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) );
   }

   vm->referenceItem( vm->regA(), *itm );
}

/*#
   @method globals Module
   @brief Returns the list of global symbols available in this module.
   @return An array containing all the global symbols.

   This method returns an array containing a list of strings, each one
   representing a name of a global symbol exposed by this module.

   The symbol names can be then fed in @a Module.set and @a Module.get methods
   to manipulate the symbols in the module.
*/
FALCON_FUNC Module_globals( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   const SymbolTable *symtab = &modc->liveModule()->module()->symbolTable();
   CoreArray* ret = new CoreArray( symtab->size() );
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      if ( ! sym->isUndefined() )
         ret->append( new CoreString(sym->name()) );
      // next symbol
      iter.next();
   }

   vm->retval( ret );
}

/*#
   @method exported Module
   @brief Returns the list of exported symbols available in this module.
   @return An array containing all the exported symbols.

   This method returns an array containing a list of strings, each one
   representing a name of a global symbol exported by this module.

   The symbol names can be then fed in @a Module.set and @a Module.get methods
   to manipulate the symbols in the module.

   Notice that exported symbols are ignored by the module loader; they
   are used by the Virtual Machine to fulfil @b load requests, but this
   doesn't imply that they are honoured in every case.

*/
FALCON_FUNC Module_exported( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   const SymbolTable *symtab = &modc->liveModule()->module()->symbolTable();
   CoreArray* ret = new CoreArray( symtab->size() );
   MapIterator iter = symtab->map().begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      if ( sym->exported() )
         ret->append( new CoreString(sym->name()) );

      // next symbol
      iter.next();
   }

   vm->retval( ret );
}

/*#
   @method unload Module
   @brief Removes the module from the running virtual machine.
   @return True on success, false on failure.

   Unloads the module, eventually destroying it when there aren't
   other VMs referencing the module.

   To actually unload the module, it is necessary not to hold any
   reference to items served by the given module (functions, classes,
   objects and so on). Strings are copied locally, so they can still
   exist when the module is unloaded.
*/
FALCON_FUNC Module_unload( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   // unlink
   vm->regA().setBoolean( vm->unlink( modc->module() ) );
}

/*#
   @method engineVersion Module
   @brief Returns the version numbers of the compilation engine.
   @return A three element array containing version, subversion and patch number.

   Returns an array of three numeric elements, indicating the version number
   of the engine under which the given module was compiled. The value is available
   both for binary modules and for pre-compiled Falcon modules. Modules compiled on
   the fly will report the same version number of the running Virtual Machine.
*/
FALCON_FUNC Module_engineVersion( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      throw new AccessError( ErrorParam( FALCOMP_ERR_UNLOADED, __LINE__ ).
         desc( FAL_STR( cmp_msg_unloaded ) ) );
   }

   const Module *mod = modc->module();

   int major, minor, re;
   mod->getEngineVersion( major, minor, re );
   CoreArray *ca = new CoreArray( 3 );
   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) re );
   vm->retval( ca );
}

/*#
   @method moduleVersion Module
   @brief Returns the version numbers of the module.
   @return A three element array containing version, subversion and patch number.

   Returns the module version information in a three element array.
   The numbers represent the development status of the module as its developers
   advertise it.

   Version informations for scripts compiled on the fly and for .fam modules
   are provided through the "version" directive.
*/
FALCON_FUNC Module_moduleVersion( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );
   const Module *mod = modc->module();

   int major, minor, re;
   mod->getModuleVersion( major, minor, re );
   CoreArray *ca = new CoreArray( 3 );
   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) re );
   vm->retval( ca );
}

/*#
   @method attributes Module
   @brief Gets the decoration attributes of this module.
   @return A dictionary with the attributes or nil if the module provides none.
*/
FALCON_FUNC Module_attributes( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );
   const Module *mod = modc->module();
   AttribMap* attr = mod->attributes();

   if ( attr == 0)
      return;

   MapIterator iter = attr->begin();
   LinearDict* cd = new LinearDict( attr->size() );
   while( iter.hasCurrent() )
   {
      VarDef* vd = *(VarDef**) iter.currentValue();

      Item itm;
      switch( vd->type() )
      {
         case VarDef::t_bool: itm.setBoolean( vd->asBool() ); break;
         case VarDef::t_int: itm.setInteger( vd->asInteger() ); break;
         case VarDef::t_num: itm.setNumeric( vd->asNumeric() ); break;
         case VarDef::t_string:
         {
            itm.setString( new CoreString( *vd->asString() ) );
         }
         break;

         default:
            itm.setNil();
      }

      cd->put( new CoreString(
         *(String*) iter.currentKey() ),
         itm
         );
      iter.next();
   }

   vm->retval( new CoreDict(cd) );
}


}
}


/* end of compiler_ext.cpp */

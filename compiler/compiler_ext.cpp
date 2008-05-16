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
#include <falcon/stringstream.h>

#include "compiler_ext.h"
#include "compiler_mod.h"

/*#
   @beginmodule feathers_compiler
*/

namespace Falcon {

namespace Ext {

/*#
   @class Compiler
   @brief Main interface to the reflexive compiler.
   @optparam path The default search path of the compiler.

   Although a single compiler should be enough for the needs of a simple script
   program, it is possible to create as many instances of the compiler as needed.

   However, it is sensible to instance this class through singleton objects,
   so that they get prepared by the link step of the VM:

   @code
      load compiler

      object MyCompiler from Compiler
      end

   @endcode

   @prop alwaysRecomp If true, a load method finding a valid .fam that may
       substitute a .fal will ignore it, and will try to compile and
       load the .fal instead.

   @prop compileInMemory If true (the default) intermediate compilation
      steps are performed in memory. If false, temporary files are used instead.

   @prop ignoreSources If true, sources are ignored, and only .fam or
      shared object/dynamic link libraries will be loaded.

   @prop path The search path for modules loaded by name. It's a set of
      Falcon format paths (forward slashes to separate dirs, e.g. “C:/my/path”),
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

   @prop language Language code used to load language-specific string tables.
      When this entry is valorized to a valid international language code, as i.e.
      "en_US", the compiler tries to use .ftr files found besides their modules to
      alter their string tables, changing the original strings with their
      translation for the desired language. If the language table file or the
      required translation is not available, the operation silently fails and
      the module is loaded with the string untranslated.
*/

/*#
   @init Compiler
   @brief Initializes the compiler with a default path.
   If @b path is not provided, defaults to “.” (script current working directory).

*/
FALCON_FUNC Compiler_init( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param( 0 );

   CompilerIface *iface;

   if( i_path != 0 )
   {
      if( ! i_path->isString() )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "[S]" ) ) );
         return;
      }

      iface = new CompilerIface( vm->self().asObject(), *i_path->asString() );
   }
   else
      iface = new CompilerIface( vm->self().asObject() );

   // set our VM as the error handler for this loader.
   iface->loader().errorHandler( vm );

   vm->self().asObject()->setUserData( iface );
}


/*#
   @method addFalconPath Compiler
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
FALCON_FUNC Compiler_addFalconPath( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );
   iface->loader().addFalconPath();

}

void internal_link( ::Falcon::VMachine *vm, Module *mod, CompilerIface *iface )
{

   Runtime rt( &iface->loader(), vm );

   // let's try to link
   if ( ! rt.addModule( mod ) || ! vm->link( &rt ) )
   {
      // VM should have raised the errors.
      mod->decref();
      vm->retnil();
      return;
   }

   // ok, the module is up and running.
   // wrap it
   Item *mod_class = vm->findWKI( "Module" );
   fassert( mod_class != 0 );
   CoreObject *co = mod_class->asClass()->createInstance();
   // we know the module IS in the VM.
   co->setUserData( new ModuleCarrier( vm->findModule( mod->name() ) ) );

   co->setProperty( "name", mod->name() );
   co->setProperty( "path", mod->path() );

   // return the object
   vm->retval( co );

   // we can remove our reference
   //mod->decref();
}

/*#
   @method compile Compiler
   @brief Compiles a script on the fly.
   @param modName A logical unique that will be given to the module after compilation.
   @param data The data to compile. It may be a string or a stream valid for input.
   @return On success, a @a Module instance that contains the compiled module.
   @raise SyntaxError if the module contains logical srror.
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S, S|O" ) ) );
      return;
   }

   Stream *input;
   String name;
   bool bDelete;

   // now, if data is an object it must be a stream.
   if( i_data->isObject() )
   {
      CoreObject *data = i_data->asObject();
      if ( ! data->derivedFrom( "Stream" ) )
      {
         vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).
            extra( "Object must be a stream" ) ) );
         return;
      }

      // ok, get the stream
      input = (Stream *) data->getUserData();
      name = "unknown_module";
      bDelete = false;
   }
   else {
      // if it's a string, we have to create a stream
      name = *i_data->asString();
      input = new StringStream( name );
      bDelete = true;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Module *mod = iface->loader().loadSource( input, name );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
   {
      mod->name( *i_name->asString() );
      internal_link( vm, mod, iface );
   }

   if( bDelete )
      delete input;
}

/*#
   @method loadByName Compiler
   @brief Loads a module given its logical name.
   @param modName The logical name of the module to be loaded.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical srror.
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Symbol *caller_sym;
   const Module *caller_mod;
   String modname;
   if ( vm->getCaller( caller_sym, caller_mod ) )
      modname = caller_mod->name();

   Module *mod = iface->loader().loadName( *i_name->asString(), modname );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
      internal_link( vm, mod, iface );
}

/*#
   @method loadModule Compiler
   @brief Loads a Falcon resource from a location on the filesystem.
   @param modPath Relative or absolute path to a loadable Falcon module or source.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical srror.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Loads the given file, trying to perform compilation or loading of the relevant
   .fam precompiled module depending on the property settings. In example, if
   loading “./test.fal”, unless alwaysRecomp property is true, “./test.fam” will be
   searched too, and if it's found and newer than ./test.fal, it will be loaded
   instead, skipping compilation step. Similarly, if “./test.fam” is searched,
   unless ignoreSource is true, “./test.fal” will be searched too, and if it's newer
   than ./test.fam it will be recompiled.

   In case a suitable module cannot be found, the method returns nil. If a module is found,
   a CodeError is raised in case compilation or link steps fails.
*/
FALCON_FUNC Compiler_loadModule( ::Falcon::VMachine *vm )
{
   Item *i_name = vm->param( 0 );

   if( i_name == 0 || ! i_name->isString() )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );

   Module *mod = iface->loader().loadFile( *i_name->asString() );

   // if mod is zero, do nothing: vm has already raised the error.
   if ( mod != 0 )
      internal_link( vm, mod, iface );
}


/*#
   @method setDirective Compiler
   @brief Compiles a script on the fly.
   @param dt Directive to be set.
   @param value Value to be given to the directive.
   @return On success, a @a Module instance that contains the loaded module.
   @raise SyntaxError if the module contains logical srror.
   @raise IoError if the input data is a file stream and there have been a read failure.

   Sets a directive as if the scripts that will be loaded by this compiler defined it
   through the directive statement. Scripts can always override a directive by setting
   it to a different value in their code; notice also that compilation directives are
   useful only if a compilation actually take places. In case a .fam or a binary module
   is loaded, they have no effect.
*/

FALCON_FUNC Compiler_setDirective( ::Falcon::VMachine *vm )
{
   Item *i_directive = vm->param( 0 );
   Item *i_value = vm->param( 1 );

   if( i_directive == 0 || ! i_directive->isString() ||
       i_value == 0 || ( ! i_value->isString() && ! i_value->isOrdinal() ) )
   {
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,S|N" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   CompilerIface *iface = static_cast<CompilerIface *>( self->getUserData() );
   if ( i_value->isString() )
      iface->loader().compiler().setDirective( *i_directive->asString(), *i_value->asString() );
   else
      iface->loader().compiler().setDirective( *i_directive->asString(), i_value->forceInteger() );

   // in case of problems, an error is already raised.
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new AccessError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S,X" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0|| ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new AccessError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
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
      vm->raiseModError( new ParamError( ErrorParam( e_inv_params, __LINE__ ).extra( "S" ) ) );
      return;
   }

   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( modc == 0 || ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new AccessError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   Item *itm = modc->liveModule()->findModuleItem( *i_name->asString() );
   if( itm == 0 )
   {
      vm->raiseModError( new AccessError( ErrorParam( e_undef_sym, __LINE__ ).
         extra(*i_name->asString()) ) );
      return;
   }

   vm->referenceItem( vm->regA(), *itm );
}

/*#
   @method unload Module
   @brief Removes the module from the running virtual machine.
   @return True on success, false on failure.

   Unloads the module, eventually destroying it when there aren't
   other VMs referencing the module.

   References to callable items that resided in the module becomes
   “ghost”. They are turned into nil when trying to use them or
   when the garbage collector reaches them; so, trying to call a
   function that resided in an unloaded module has the same effect
   as calling a nil item, raising an error.
*/
FALCON_FUNC Module_unload( ::Falcon::VMachine *vm )
{
   CoreObject *self = vm->self().asObject();
   ModuleCarrier *modc = static_cast<ModuleCarrier *>( self->getUserData() );

   // if the module is not alive, raise an error and exit
   if ( ! modc->liveModule()->isAlive() )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new AccessError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   // unlink
   if ( vm->unlink( modc->module() ) )
   {

      // destroy the reference
      delete modc;
      self->setUserData( 0 );

      // report success.
      vm->regA().setBoolean( true );
   }
   else {
      vm->regA().setBoolean( false );
   }
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
   if ( modc == 0 )
   {
      // TODO: Find a more adequate error code.
      vm->raiseModError( new AccessError( ErrorParam( e_modver, __LINE__ ) ) );
      return;
   }

   const Module *mod = modc->module();

   int major, minor, re;
   mod->getEngineVersion( major, minor, re );
   CoreArray *ca = new CoreArray( vm, 3 );
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
   CoreArray *ca = new CoreArray( vm, 3 );
   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) re );
   vm->retval( ca );
}

}
}


/* end of compiler_ext.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: vminfo_ext.h

   Header for Falcon Realtime Library - C modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 14 Aug 2008 00:31:21 +0200

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include "core_module.h"
#include <falcon/stackframe.h>
#include <falcon/sys.h>

namespace Falcon {
namespace core {

/*#
   @funset vminfo Virtual Machine Informations
   @brief Generic informations on the Virtual Machine.

   This functions are meant to provide minimal informations about the
   virtual machine and its configuration. For example, they provide
   the VM version number and target architectures.
*/

/*#
   @function vmVersionInfo
   @inset vminfo
   @brief Returns an array containing VM version informations.
   @return Major, minor and revision numbers of the running virtual machine in a 3 elements array.
*/
FALCON_FUNC  vmVersionInfo( ::Falcon::VMachine *vm )
{
   CoreArray *ca = new CoreArray( 3 );
   ca->append( (int64) ((FALCON_VERSION_NUM >> 16)) );
   ca->append( (int64) ((FALCON_VERSION_NUM >> 8) & 0xFF) );
   ca->append( (int64) ((FALCON_VERSION_NUM ) & 0xFF) );
   vm->retval( ca );
}

/*#
   @function vmModuleVersionInfo
   @inset vminfo
   @brief Returns an array containing current module version informations.
   @return Major, minor and revision numbers of the curerntly being executed module,
      in a 3 elements array.
*/
FALCON_FUNC  vmModuleVersionInfo( ::Falcon::VMachine *vm )
{
   CoreArray *ca = new CoreArray( 3 );
   int major=0, minor=0, revision=0;

   // we don't want our current (core) module version info...
   StackFrame *thisFrame = vm->currentFrame();
   if( thisFrame->prev() != 0 )
   {
      StackFrame *prevFrame = thisFrame->prev();
      if ( prevFrame->m_module != 0 )
      {
         prevFrame->m_module->module()->getModuleVersion( major, minor, revision );
      }
   }

   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) revision );
   vm->retval( ca );
}

/*#
   @function vmVersionName
   @inset vminfo
   @brief Returns the nickname for this VM version.
   @return A string containing the symbolic name of this VM version.
*/
FALCON_FUNC  vmVersionName( ::Falcon::VMachine *vm )
{
   CoreString *str = new CoreString(  FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   vm->retval( str );
}

/*#
   @function vmSystemType
   @inset vminfo
   @brief Returns a descriptive name of the overall system architecture.
   @return A string containing a small descriptiuon of the system architecture.

   Currently, it can be "WIN" on the various MS-Windows flavours and POSIX on
   Linux, BSD, Solaris, Mac-OSX and other *nix based systems.
*/
FALCON_FUNC  vmSystemType( ::Falcon::VMachine *vm )
{
   CoreString *str = new CoreString(  Sys::SystemData::getSystemType() );
   vm->retval( str );
}

/*#
   @function vmIsMain
   @inset vminfo
   @brief Returns true if the calling module is the main module of the application.
   @return True if the calling module is the main module.

   This function checks if the current module has been added as the last one right
   before starting an explicit execution of the virtual machine from the outside.

   This function is useful for those modules that have a main code which is meant
   to be executed at link time and a part that is menat to be executed only if the
   module is directly loaded and executed.

   For example:
   @code
      // executes this at link time
      prtcode = printl

      // executes this from another module on request
      function testPrint()
         prtcode( "Success." )
      end
      export testPrint

      // performs a test if directly loaded
      if vmIsMain()
         > "Testing the testPrint function"
         testPrint()
      end
   @endcode

*/
FALCON_FUNC vmIsMain( ::Falcon::VMachine *vm )
{
   StackFrame *thisFrame = vm->currentFrame();
   if ( thisFrame == 0 )
   {
      throw new GenericError( ErrorParam( e_stackuf, __LINE__ ).origin( e_orig_runtime ) );
   }
   else {
      // get the calling symbol module
      vm->retval( (bool) (thisFrame->m_module == vm->mainModule() ) );
   }
}

/*#
   @function vmFalconPath
   @inset vminfo
   @brief Returns default system path for Falcon load requests.
   @return The default compiled-in load path, or the value of the
      environemnt variable FALCON_LOAD_PATH if defined.
*/

FALCON_FUNC vmFalconPath( ::Falcon::VMachine *vm )
{
   String envpath;
   bool hasEnvPath = Sys::_getEnv( "FALCON_LOAD_PATH", envpath );

   if ( hasEnvPath )
   {
      vm->retval( new CoreString(  envpath ) );
   }
   else {
      vm->retval( new CoreString(  FALCON_DEFAULT_LOAD_PATH ) );
   }
}


/*#
   @function vmSearchPath
   @inset vminfo
   @brief Returns the application specific load path.
   @return A module search path as set by the application when creating the virtual machine.
   
   This string is at disposal of the embeddign application (or of the Falcon command line
   interpreter) to communicate to scripts and underlying users the search path set at
   applicaiton level. It is used by internal services, the @a include function, the
   compiler Feather module and similar facilities.
*/

FALCON_FUNC vmSearchPath( ::Falcon::VMachine *vm )
{
   vm->retval( new CoreString( vm->appSearchPath() ) );
}

/*#
   @function vmModuleName
   @inset vminfo
   @brief Returns the logical name of this module.
   @return Logical name of this module.

   Every module has a logical name inside the application.
   There is a loose mapping between the underlying module providers
   and the logical name of the module, so knowing it may be helpful.
*/

FALCON_FUNC vmModuleName( ::Falcon::VMachine *vm )
{
   const Symbol* sym;
   const Module* mod;

   vm->getCaller( sym, mod );
   vm->retval( new CoreString( mod->name() ));
}

/*
   @function vmModulePath
   @inset vminfo
   @brief Returns the phisical path (complete URI) from which this module was loaded.
   @return A string representing the load URI that individuates this module.
*/
FALCON_FUNC vmModulePath( ::Falcon::VMachine *vm )
{
   const Symbol* sym;
   const Module* mod;

   vm->getCaller( sym, mod );
   vm->retval( new CoreString( mod->path() ));
}

/*
   @function vmRelativePath
   @inset vminfo
   @param path A relative (or absolute) path to a target file.
   @brief Relativize the URI of a given file to the path of the current module.
   @return A string containing the relativized URI.

   This function is meant to simplify the task of loading dynamic components
   in falcon libraries. The static loader has enough information to form
   a correct script dependency tree, but a library may be placed in unexpected
   loactions at target site, and then it would be complex to dynamcally load
   components. Suppose a library has a module called plugin.fal in the subdirectory
   modules/ of its directory structure. To load it, this function can be used
   to virtualize the load path so that
   @code
      modpath = vmRelativePath( "modules/plugin.fal" )
      include( modpath )
   @endcode
*/
FALCON_FUNC vmRelativePath( ::Falcon::VMachine *vm )
{
   Item *i_path = vm->param(0);
   if( i_path == 0 || ! i_path->isString())
   {
      throw new Falcon::ParamError(
         Falcon::ErrorParam( Falcon::e_inv_params, __LINE__ ).
         extra( "S" ) );
   }

   const Symbol* sym;
   const Module* mod;

   Path path( *i_path->asString() );

   vm->getCaller( sym, mod );
   URI uri( mod->path() );
   if ( path.isAbsolute() )
      uri.pathElement().setLocation( path.getLocation() );
   else if ( path.getLocation() != "" )
      uri.pathElement().extendLocation( path.getLocation() );

   uri.pathElement().setFilename( path.getFilename() );
   CoreString* ret = new CoreString( uri.get(false) );
   ret->bufferize();
   vm->retval( ret );
}

}
}

/* end of vminfo_ext.cpp */

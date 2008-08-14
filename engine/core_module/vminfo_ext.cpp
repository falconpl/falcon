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

#include <falcon/engine.h>
#include "core_module.h"

namespace Falcon {
namespace core {

/*#
   @funset vminfo Virtual Machine Informations
   @brief Generic informations on the Virtual Machine.

   This functions are meant to provide minimal informations about the
   virtual machine and its configuration. In example, they provide
   the VM version number and target architectures.
*/

/*#
   @function vmVersionInfo
   @ingroup vminfo
   @inset vminfo
   @brief Returns an array containing VM version informations.
   @return Major, minor and revision numbers of the running virtual machine in a 3 elements array.
*/
FALCON_FUNC  vmVersionInfo( ::Falcon::VMachine *vm )
{
   CoreArray *ca = new CoreArray( vm, 3 );
   ca->append( (int64) ((FALCON_VERSION_NUM >> 16)) );
   ca->append( (int64) ((FALCON_VERSION_NUM >> 8) & 0xFF) );
   ca->append( (int64) ((FALCON_VERSION_NUM ) & 0xFF) );
   vm->retval( ca );
}

/*#
   @function vmModuleVersionInfo
   @ingroup vminfo
   @inset vminfo
   @brief Returns an array containing current module version informations.
   @return Major, minor and revision numbers of the curerntly being executed module,
      in a 3 elements array.
*/
FALCON_FUNC  vmModuleVersionInfo( ::Falcon::VMachine *vm )
{
   CoreArray *ca = new CoreArray( vm, 3 );
   int major=0, minor=0, revision=0;

   // we don't want our current (core) module version info...
   StackFrame *thisFrame = (StackFrame *) &vm->stackItem( vm->stackBase() - VM_FRAME_SPACE );
   if( thisFrame->m_stack_base != 0 )
   {
      StackFrame *prevFrame = (StackFrame *) &vm->stackItem( thisFrame->m_stack_base - VM_FRAME_SPACE );
      if ( prevFrame->m_module != 0 )
      {
         prevFrame->m_module->getModuleVersion( major, minor, revision );
      }
   }

   ca->append( (int64) major );
   ca->append( (int64) minor );
   ca->append( (int64) revision );
   vm->retval( ca );
}

/*#
   @function vmVersionName
   @ingroup vminfo
   @inset vminfo
   @brief Returns the nickname for this VM version.
   @return A string containing the symbolic name of this VM version.
*/
FALCON_FUNC  vmVersionName( ::Falcon::VMachine *vm )
{
   String *str = new GarbageString( vm, FALCON_VERSION " (" FALCON_VERSION_NAME ")" );
   vm->retval( str );
}

/*#
   @function vmSystemType
   @ingroup vminfo
   @inset vminfo
   @brief Returns a descriptive name of the overall system architecture.
   @return A string containing a small descriptiuon of the system architecture.

   Currently, it can be "WIN" on the various MS-Windows flavours and POSIX on
   Linux, BSD, Solaris, Mac-OSX and other *nix based systems.
*/
FALCON_FUNC  vmSystemType( ::Falcon::VMachine *vm )
{
   String *str = new GarbageString( vm, Sys::SystemData::getSystemType() );
   vm->retval( str );
}

}
}

/* end of vminfo_ext.cpp */

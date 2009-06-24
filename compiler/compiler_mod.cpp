/*
   FALCON - The Falcon Programming Language
   FILE: compiler_mod.cpp

   Compiler interface modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab lug 21 2007

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Compiler interface modules
*/

#include <falcon/module.h>
#include <falcon/coreobject.h>
#include <falcon/vm.h>
#include <falcon/stream.h>

#include "compiler_mod.h"

namespace Falcon {

namespace Ext {

CoreObject* CompilerIfaceFactory( const CoreClass *cls, void *, bool )
{
   return new CompilerIface(cls);
}

// Implemented here to reduce inline overhead
CompilerIface::CompilerIface( const CoreClass *cls ):
   CoreObject( cls ),
   m_loader( "." ),
   m_bLaunchAtLink( false )
{
   // get default source encoding
   m_sourceEncoding = m_loader.sourceEncoding();
   m_loader.delayRaise( true );
}

// Implemented here to reduce inline overhead
CompilerIface::CompilerIface(  const CoreClass *cls, const String &path ):
   CoreObject( cls ),
   m_loader( path ),
   m_bLaunchAtLink( false )
{
   // get default source encoding
   m_sourceEncoding = m_loader.sourceEncoding();
   m_loader.delayRaise( true );
}


// Implemented here to reduce inline overhead
CompilerIface::~CompilerIface()
{}


bool CompilerIface::getProperty( const String &propName, Item &prop ) const
{
   if( propName == "path" )
   {
      if ( ! prop.isString() )
         prop = new CoreString();

      m_loader.getSearchPath( *prop.asString() );
   }
   else if( propName == "alwaysRecomp" )
   {
      prop.setBoolean( m_loader.alwaysRecomp() );
   }
   else if( propName == "compileInMemory" )
   {
      prop.setBoolean( m_loader.compileInMemory() );
   }
   else if( propName == "ignoreSources" )
   {
      prop.setBoolean( m_loader.ignoreSources() );
   }
   else if( propName == "saveModules" )
   {
      prop.setBoolean( m_loader.saveModules() );
   }
   else if( propName == "saveMandatory" )
   {
      prop.setBoolean( m_loader.saveMandatory() );
   }
   else if( propName == "sourceEncoding" )
   {
      prop = new CoreString( m_loader.sourceEncoding() );
   }
   else if( propName == "detectTemplate" )
   {
      prop.setBoolean( m_loader.saveMandatory() );
   }
   else if( propName == "compileTemplate" )
   {
      prop.setBoolean( m_loader.saveMandatory() );
   }
   else if( propName == "launchAtLink" )
   {
      prop.setBoolean( m_bLaunchAtLink );
   }
   else if( propName == "langauge" )
   {
      if ( ! prop.isString() )
         prop = new CoreString;
      *prop.asString() = m_loader.getLanguage();
   }
   else {
      return defaultProperty( propName, prop );
   }

   return true;
}


bool CompilerIface::setProperty( const String &propName, const Item &prop )
{
   if( propName == "path" && prop.isString() )
   {
      m_loader.setSearchPath( *prop.asString() );
   }
   else if( propName == "language" && prop.isString() )
   {
      m_loader.setLanguage( *prop.asString() );
   }
   else if( propName == "alwaysRecomp" )
   {
      m_loader.alwaysRecomp( prop.isTrue() );
   }
   else if( propName == "compileInMemory" )
   {
      m_loader.compileInMemory( prop.isTrue() );
   }
   else if( propName == "ignoreSources" )
   {
      m_loader.ignoreSources( prop.isTrue() );
   }
   else if( propName == "saveModules" )
   {
      m_loader.saveModules( prop.isTrue() );
   }
   else if( propName == "saveMandatory" )
   {
      m_loader.saveMandatory( prop.isTrue() );
   }
   else if( propName == "sourceEncoding" && prop.isString() )
   {
      m_loader.sourceEncoding( *prop.asString() );
   }
   else if( propName == "detectTemplate" )
   {
      m_loader.detectTemplate( prop.isTrue() );
   }
   else if( propName == "compileTemplate" )
   {
      m_loader.compileTemplate( prop.isTrue() );
   }
   else if( propName == "launchAtLink" )
   {
      m_bLaunchAtLink = prop.isTrue();
   }
   else {
      readOnlyError( propName );
   }

   return true;
}


ICompilerIface::ICompilerIface( const CoreClass* cls ):
   CompilerIface( cls )
{
   m_vm = new VMachine;
   m_intcomp = new InteractiveCompiler( &m_loader, m_vm );
}

ICompilerIface::ICompilerIface( const CoreClass* cls, const String &path ):
   CompilerIface( cls, path )
{
   m_vm = new VMachine;
}

ICompilerIface::~ICompilerIface()
{
   delete m_intcomp;
   m_vm->finalize();
}


bool ICompilerIface::setProperty( const String &prop, const Item &value )
{
   if( prop == "stdIn" && value.isObject() && value.asObjectSafe()->derivedFrom("Stream") )
   {
      Stream *clone = static_cast<Stream *>( value.asObjectSafe()->getFalconData()->clone());
      m_vm->stdIn( clone );
   }
   else if( prop == "stdOut" && value.isObject() && value.asObjectSafe()->derivedFrom("Stream") )
   {
      Stream *clone = static_cast<Stream *>( value.asObjectSafe()->getFalconData()->clone());
      m_vm->stdOut( clone );
   }
   else if( prop == "stdErr" && value.isObject() && value.asObjectSafe()->derivedFrom("Stream") )
   {
      Stream *clone = static_cast<Stream *>( value.asObjectSafe()->getFalconData()->clone());
      m_vm->stdErr( clone );
   }

   return CompilerIface::setProperty( prop, value );
}

bool ICompilerIface::getProperty( const String &prop, Item &ret ) const
{
   Stream *s = 0;

   if( prop == "stdIn" )
   {
      s = m_vm->stdIn();
   }
   else if( prop == "stdOut" )
   {
      s = m_vm->stdOut();
   }
   else if( prop == "stdErr" )
   {
      s = m_vm->stdErr();
   }
   else if( prop == "value" )
   {
      ret = m_vm->regA();
      return true;
   }
   else {
      return CompilerIface::getProperty( prop, ret );
   }

   // s here is initialized
   fassert( s != 0 );

   Item* stream_class = VMachine::getCurrent()->findWKI( "StdStream" );
   fassert( stream_class != 0 );
   CoreObject *co = stream_class->asClass()->createInstance( s->clone() );
   ret = co;

   return true;
}


//=========================================================



ModuleCarrier::ModuleCarrier( LiveModule *module ):
   m_lmodule( module )
{
}

ModuleCarrier::~ModuleCarrier()
{
   // the LiveModule does not belong to us, and by this time it may be already gone
}

FalconData *ModuleCarrier::clone() const
{
   return new ModuleCarrier( m_lmodule );
}

void ModuleCarrier::gcMark( uint32 mk )
{
   m_lmodule->mark( mk );
}


const Module *ModuleCarrier::module() const
{
   return m_lmodule->module();
}

}
}

/* end of compiler_mod.cpp */

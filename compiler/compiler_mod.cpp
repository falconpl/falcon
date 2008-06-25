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
#include <falcon/cobject.h>
#include <falcon/vm.h>
#include "compiler_mod.h"

namespace Falcon {

namespace Ext {

// Implemented here to reduce inline overhead
CompilerIface::CompilerIface( CoreObject *owner ):
   m_loader( "." ),
   m_owner( owner )
{
   // get default source encoding
   m_sourceEncoding = m_loader.sourceEncoding();
   m_loader.delayRaise( true );
}

// Implemented here to reduce inline overhead
CompilerIface::CompilerIface( CoreObject *owner, const String &path ):
   m_loader( path ),
   m_owner( owner )
{
   // get default source encoding
   m_sourceEncoding = m_loader.sourceEncoding();
   m_loader.delayRaise( true );
}


// Implemented here to reduce inline overhead
CompilerIface::~CompilerIface()
{}

void CompilerIface::getProperty( VMachine *, const String &propName, Item &prop )
{
   if( propName == "path" )
   {
      if ( ! prop.isString() )
         prop = new GarbageString( m_owner->origin() );
      m_loader.getSearchPath( *prop.asString() );
   }
   else if( propName == "alwaysRecomp" )
   {
      prop = (int64) ( m_loader.alwaysRecomp() ? 1: 0 );
   }
   else if( propName == "compileInMemory" )
   {
      prop = (int64) ( m_loader.compileInMemory() ? 1: 0 );
   }
   else if( propName == "ignoreSources" )
   {
      prop = (int64) ( m_loader.ignoreSources() ? 1: 0 );
   }
   else if( propName == "saveModules" )
   {
      prop = (int64) ( m_loader.saveModules() ? 1: 0 );
   }
   else if( propName == "saveMandatory" )
   {
      prop = (int64) ( m_loader.saveMandatory() ? 1: 0 );
   }
   else if( propName == "sourceEncoding" )
   {
      prop = new GarbageString( m_owner->origin(), m_loader.sourceEncoding() );
   }
   else if( propName == "detectTemplate" )
   {
      prop = (int64) ( m_loader.saveMandatory() ? 1: 0 );
   }
   else if( propName == "compileTemplate" )
   {
      prop = (int64) ( m_loader.saveMandatory() ? 1: 0 );
   }
   else if( propName == "langauge" )
   {
      if ( ! prop.isString() )
         prop = new GarbageString( m_owner->origin() );
      *prop.asString() = m_loader.getLanguage();
   }
}

void CompilerIface::setProperty( VMachine *, const String &propName, const Item &prop )
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

void ModuleCarrier::gcMark( VMachine *vm )
{
   m_lmodule->mark( vm->memPool()->currentMark() );
}

}
}

/* end of compiler_mod.cpp */

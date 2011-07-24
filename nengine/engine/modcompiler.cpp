/*
   FALCON - The Falcon Programming Language.
   FILE: modcompiler.cpp

   Interactive compiler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 07 May 2011 16:04:05 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/modcompiler.h>
#include <falcon/module.h>

#include "falcon/exprvalue.h"

namespace Falcon {

ModCompiler::Context::Context( ModCompiler* owner ):
   m_owner(owner),
   m_nLambdaCount(0),
   m_nClsCount(0)
{
}

ModCompiler::Context::~Context()
{

}

void ModCompiler::Context::onInputOver()
{

}

void ModCompiler::Context::onNewFunc( Function* function, GlobalSymbol* gs )
{
   if( gs == 0 )
   {
      // anonymous function
      String name = "__lambda#";
      name.N(m_nLambdaCount++);
      function->name( name );
      m_owner->m_module->addFunction( function, false );
   }
   else
   {
      m_owner->m_module->addFunction( gs, function );
   }
}


void ModCompiler::Context::onNewClass( Class* cls, bool bIsObj, GlobalSymbol* gs )
{
   if( gs == 0 )
   {
      // anonynmous class
      String name = "__class#";
      name.N(m_nClsCount++);
      cls->name( name );
      m_owner->m_module->addClass( cls );
   }
}


void ModCompiler::Context::onNewStatement( Statement* stmt )
{

}


void ModCompiler::Context::onLoad( const String& path, bool isFsPath )
{

}


void ModCompiler::Context::onImportFrom( const String& path, bool isFsPath, const String& symName,
      const String& asName, const String &inName )
{

}


void ModCompiler::Context::onImport(const String& symName )
{

}


void ModCompiler::Context::onExport(const String& symName)
{

}


void ModCompiler::Context::onDirective(const String& name, const String& value)
{

}


void ModCompiler::Context::onGlobal( const String& name )
{

}


Symbol* ModCompiler::Context::onUndefinedSymbol( const String& name )
{

}


GlobalSymbol* ModCompiler::Context::onGlobalDefined( const String& name, bool& bUnique )
{

}


bool ModCompiler::Context::onUnknownSymbol( UnknownSymbol* sym )
{

}


Expression* ModCompiler::Context::onStaticData( Class* cls, void* data )
{
   m_owner->m_module->addStaticData( cls, data );

   if( m_owner->m_module->isStatic() )
   {
      return new ExprValue( Item( cls, data ) );
   }
   else
   {
      return new ExprValue( Item( cls, data, true ) );
   }
}


void ModCompiler::Context::onInheritance( Inheritance* inh  )
{
   
}

//=================================================================
//
//

ModCompiler::ModCompiler():
   m_module(0)
{
   m_ctx = new Context( this );
}

ModCompiler::~ModCompiler()
{
   delete m_ctx;
}


Module* compile( TextReader* input, const String& uri, const String& name )
{
}

}

/* end of modcompiler.cpp */

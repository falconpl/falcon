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
#include <falcon/falconclass.h>
#include <falcon/hyperclass.h>
#include <falcon/exprvalue.h>
#include <falcon/globalsymbol.h>
#include <falcon/inheritance.h>

namespace Falcon {

ModCompiler::Context::Context( ModCompiler* owner ):
   ParserContext( &owner->m_sp ),
   m_owner(owner)
{
}

ModCompiler::Context::~Context()
{

}

void ModCompiler::Context::onInputOver()
{
   // todo: check undefined things.
}

void ModCompiler::Context::onNewFunc( Function* function, GlobalSymbol* gs )
{
   if( gs == 0 )
   {
      // anonymous function
      String name = "__lambda#";
      name.N( m_owner->m_nLambdaCount++);
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
      name.N( m_owner->m_nClsCount++ );
      cls->name( name );
   }
   
   FalconClass* fcls = static_cast<FalconClass*>(cls);
   if( !fcls->construct() )
   {
      // did we fail to construct because we're incomplete?
      if( ! fcls->missingParents() )
      {
         // so, we have to generate an hyper class out of our falcon-class
         // -- the hyperclass is also owning the FalconClass.
         Class* hyperclass = fcls->hyperConstruct();
         m_owner->m_module->addClass( gs, hyperclass, bIsObj );
      }
      else
      {
         // wait cfor completion in link phase
         m_owner->m_module->addClass( gs, cls, bIsObj );
      }
   }
   else
   {
      m_owner->m_module->addClass( gs, cls, bIsObj );
   }
}


void ModCompiler::Context::onNewStatement( Statement* )
{
   // actually nothing to do
}


void ModCompiler::Context::onLoad( const String& path, bool isFsPath )
{
   m_owner->m_module->addLoad( path, isFsPath );
}


void ModCompiler::Context::onImportFrom( const String& path, bool isFsPath, const String& symName,
      const String& asName, const String &inName )
{
   // TODO: create the namespace
   String localName = asName != "" ? asName : inName + "." + symName;
   m_owner->m_module->addImportFrom( localName,  symName, path, isFsPath );
}


void ModCompiler::Context::onImport(const String& symName )
{
   m_owner->m_module->addImport( symName );
}


void ModCompiler::Context::onExport(const String& symName)
{
   m_owner->m_module->addVariable( symName, true );
}


void ModCompiler::Context::onDirective(const String&, const String& )
{
   // TODO
}


void ModCompiler::Context::onGlobal( const String& )
{
   // TODO
}


Symbol* ModCompiler::Context::onUndefinedSymbol( const String& name )
{
   // Is this a global symbol?
   Symbol* gsym = m_owner->m_module->getGlobal( name );
   if( gsym == 0 )
   {
      // Then require it to be imported.
      gsym = (Symbol*) m_owner->m_module->addImport( name );
   }
   
   return gsym;
}


GlobalSymbol* ModCompiler::Context::onGlobalDefined( const String& name, bool& bAlreadyDef )
{
   Symbol* sym = m_owner->m_module->getGlobal(name);
   if( sym == 0 )
   {
      bAlreadyDef = false;
      return m_owner->m_module->addVariable( name );
   }

   bAlreadyDef = true;
   return static_cast<GlobalSymbol*>(sym);
}


bool ModCompiler::Context::onUnknownSymbol( UnknownSymbol* )
{
   // let the unknown symbol to be stored in the tree.
   return false;
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
   // In the interactive compiler context, classes must have been already defined...
   const Symbol* sym = m_owner->m_module->getGlobal(inh->className());
  
   // found?
   if( sym != 0 )
   {
      Item itm;
      if( ! sym->retrieve( itm, 0 ) )
      {
         //TODO: Add inheritance line number.
         m_owner->m_sp.addError( e_undef_sym, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      // we want a class. A rdeal class.
      else if ( ! itm.isUser() || ! itm.asClass()->isMetaClass() )
      {
         m_owner->m_sp.addError( e_inv_inherit, m_owner->m_sp.currentSource(), 
                 inh->sourceRef().line(), inh->sourceRef().chr(), 0, inh->className() );
      }
      else
      {
         inh->parent( static_cast<Class*>(itm.asInst()) );
      }

      // ok, now how to break away if we aren't complete?
   }
   else
   {
      m_owner->m_module->addImportInheritance( inh );
   }
}

//=================================================================
//
//

ModCompiler::ModCompiler():
   m_module(0),
   m_nLambdaCount(0),
   m_nClsCount(0)
{
   m_ctx = new Context( this );
}

ModCompiler::~ModCompiler()
{
   delete m_ctx;
}


Module* compile( TextReader*, const String&, const String& )
{
   return 0;
}

}

/* end of modcompiler.cpp */

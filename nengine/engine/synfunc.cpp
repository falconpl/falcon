/*
   FALCON - The Falcon Programming Language.
   FILE: synfunc.h

   SynFunc objects -- expanding to new syntactic trees.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/synfunc.h>
#include <falcon/vmcontext.h>
#include <falcon/item.h>
#include <falcon/statement.h>
#include <falcon/trace.h>
#include <falcon/localsymbol.h>
#include <falcon/closedsymbol.h>

namespace Falcon
{

SynFunc::SynFunc( const String& name, Module* owner, int32 line ):
   Function( name, owner, line )
{}


SynFunc::~SynFunc()
{
   for ( size_t i = 0; i < m_locals.size(); ++i )
   {
      delete m_locals[i];
   }
}

Symbol* SynFunc::addVariable( const String& name )
{
   Symbol* sym = new LocalSymbol( name, m_locals.size() );
   m_locals.push_back( sym );
   m_symtabTable[name] = sym;
   return sym;
}


Symbol* SynFunc::addClosedSymbol( const String& name, const Item& value )
{
   Symbol* sym = new ClosedSymbol( name, value );
   m_locals.push_back( sym );
   m_symtabTable[name] = sym;
   return sym;
}


Symbol* SynFunc::findSymbol( const String& name ) const
{
   SymbolTable::const_iterator pos = m_symtabTable.find( name );
   if( pos == m_symtabTable.end() )
   {
      return 0;
   }

   return pos->second;
}

Symbol* SynFunc::getSymbol( int32 id ) const
{
   if ( id < 0 || id > (int) m_locals.size() )
   {
      return 0;
   }

   return m_locals[id];
}


void SynFunc::apply( VMachine* vm, int32 nparams )
{
   // Used by the VM to insert this opcode if needed to exit SynFuncs.
   static StmtReturn s_a_return;

   // nothing to do?
   if( syntree().empty() )
   {
      TRACE( "-- function %s is empty -- not calling it", locate().c_ize() );
      vm->returnFrame();
      return;
   }

   register VMContext* ctx = vm->currentContext();
   
   // fill the parameters
   TRACE1( "-- filing parameters: %d/%d", nparams, this->paramCount() );
   while( nparams < this->paramCount() )
   {
      (++ctx->m_topData)->setNil();
      ++nparams;
   }

   // fill the locals
   int locals = this->varCount() - this->paramCount();
   TRACE1( "-- filing locals: %d", locals );
   while( locals > 0 )
   {
      (++ctx->m_topData)->setNil();
      --locals;
   }


   if( this->syntree().last()->type() != Statement::return_t )
   {
      TRACE1( "-- Pushing extra return", 0 );
      ctx->pushCode( &s_a_return );
   }

   ctx->pushCode( &this->syntree() );
}

}

/* end of synfunc.cpp */

/*
   FALCON - The Falcon Programming Language.
   FILE: function.cpp

   Function objects.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 Jan 2011 19:09:07 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/function.h>
#include <falcon/localsymbol.h>
#include <falcon/closedsymbol.h>
#include <falcon/item.h>
#include <falcon/collector.h>

#include <falcon/corefunction.h>

namespace Falcon
{

Function::Function( const String& name, Module* module ):
   m_name( name ),
   m_paramCount(0),
   m_gcToken( 0 ),
   m_module( 0 )
{
}

Function::~Function()
{
   for ( int i = 0; i < m_locals.size(); ++i )
   {
      delete m_locals[i];
   }
}


Symbol* Function::addVariable( const String& name )
{
   Symbol* sym = new LocalSymbol( name, m_locals.size() );
   m_locals.push_back( sym );
   m_symtabTable[name] = sym;
   return sym;
}


Symbol* Function::addClosedSymbol( const String& name, const Item& value )
{
   Symbol* sym = new ClosedSymbol( name, value );
   m_locals.push_back( sym );
   m_symtabTable[name] = sym;
   return sym;
}


Symbol* Function::findSymbol( const String& name ) const
{
   SymbolTable::const_iterator pos = m_symtabTable.find( name );
   if( pos == m_symtabTable.end() )
   {
      return 0;
   }

   return pos->second;
}

Symbol* Function::getSymbol( int32 id ) const
{
   if ( id < 0 || id > m_locals.size() )
   {
      return 0;
   }

   return m_locals[id];
}


void Function::gcMark(int32 mark)
{
   if (m_gcToken != 0 )
   {
      m_gcToken->mark(mark);
   }
}


void Function::garbage( Collector* c )
{
   m_gcToken = c->store( &CoreFunction_handler, this );
}

}

/* end of function.cpp */

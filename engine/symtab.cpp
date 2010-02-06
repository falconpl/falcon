/*
   FALCON - The Falcon Programming Language.
   FILE: symtab.cpp

   Symbol table definition
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: mar ago 3 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/symtab.h>
#include <falcon/symbol.h>
#include <falcon/stream.h>
#include <falcon/traits.h>
#include <falcon/module.h>
#include <falcon/traits.h>

namespace Falcon {

SymbolTable::SymbolTable():
   m_map( &traits::t_stringptr(), &traits::t_voidp(), 19 )
{
}

void SymbolTable::exportUndefined()
{
   MapIterator iter = m_map.begin();
   while( iter.hasCurrent() )
   {
      Symbol *sym = *(Symbol **) iter.currentValue();
      if( ! sym->isUndefined() )
         sym->exported( true );
      iter.next();
   }
}

bool SymbolTable::add( Symbol *sym )
{
   if( findByName( sym->name() ) != 0 )
      return false;

   m_map.insert( &sym->name(), sym );
   return true;
}

bool SymbolTable::remove( const String &name )
{
   return m_map.erase( &name );
}


bool SymbolTable::save( Stream *out ) const
{
   uint32 value;
   // save the symbol table size.
   value = endianInt32( size() );
   out->write( &value, sizeof(value) );
   MapIterator iter = m_map.begin();

   while( iter.hasCurrent() )
   {
      const Symbol *second = *(const Symbol **) iter.currentValue();

      value = endianInt32( second->id() );
      out->write( &value, sizeof(value) );
      iter.next();
   }

   return true;
}

bool SymbolTable::load( const Module *mod, Stream *in )
{
   // get the symtab type.
   int32 value;
   in->read( &value, sizeof(value) );
   int32 final_size = endianInt32(value);

   // preallocate all the symbols;
   for ( int i = 0 ; i < final_size; i ++ )
   {
      Symbol *sym;

      in->read( &value, sizeof(value) );
      sym = mod->getSymbol( endianInt32(value) );
      if ( sym == 0 ) {
         return false;
      }

      if ( sym->name().getRawStorage() == 0 ) {
         return false;
      }
      m_map.insert( &sym->name(), sym );
   }

   return true;
}


SymbolVector::SymbolVector():
	GenericVector( &traits::t_voidp() )
{
}

SymbolVector::~SymbolVector()
{
 // free the symbols.
   for ( uint32 i = 0; i < size(); i ++ )
   {
      delete symbolAt( i );
   }
}

bool SymbolVector::save( Stream *out ) const
{
   uint32 value = endianInt32(size());
   out->write( &value, sizeof(value) );

   for( uint32 iter = 0; iter < size(); iter++ )
   {
      symbolAt( iter )->name().serialize( out );
   }

   for( uint32 iter = 0; iter < size(); iter++ )
   {
      if ( ! symbolAt( iter )->save( out ) )
         return false;
   }

   return true;
}

bool SymbolVector::load( Module *owner, Stream *in )
{
   uint32 value;
   in->read( &value, sizeof(value) );
   value = endianInt32( value );

   resize( value );
   for ( uint32 i = 0; i < value; i ++ )
   {
      Symbol *sym = new Symbol(owner);
      sym->id( i );
      set( sym, i );
      if ( ! sym->name().deserialize( in ) )
         return false;
   }

   for( uint32 iter = 0; iter < size(); iter++ )
   {
      if ( ! symbolAt( iter )->load( in ) )
         return false;
   }

   return true;
}

}

/* end of symtab.cpp */

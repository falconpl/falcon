/*
   FALCON - The Falcon Programming Language.
   FILE: symbol.cpp
   $Id: symbol.cpp,v 1.10 2007/08/02 23:41:40 jonnymind Exp $

   Provide non-inlineizable symbol definitions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-9-11
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

#include <falcon/setup.h>
#include <falcon/symbol.h>
#include <falcon/symtab.h>
#include <falcon/module.h>
#include <falcon/stream.h>
#include <falcon/traits.h>

namespace Falcon
{

void Symbol::clear()
{
   switch( m_type )
   {
      case tvar: delete m_value.v_prop; break;
      case tfunc: delete m_value.v_func; break;
      case textfunc: delete m_value.v_extfunc; break;
      case tclass: delete m_value.v_class; break;
      case tprop: delete m_value.v_prop; break;
      case tconst: delete m_value.v_prop; break;
   }
   m_type = tundef;
}


bool Symbol::fromClass( const String &find_name ) const
{
   if( find_name == name() )
      return true;

   if( ! isClass() )
      return false;

   ClassDef *def = m_value.v_class;
   return def->inheritsFrom( find_name );
}

bool Symbol::load( Stream *in )
{
   setUndefined();
   byte type;
   in->read( &type, sizeof( type ) );
   byte exp;
   in->read( &exp, sizeof( exp ) );
   m_flags = exp;

   uint16 pos;
   in->read( &pos, sizeof( pos ) );
   m_itemPos = endianInt16( pos );

   int32 line;
   in->read( &line, sizeof( line ) );
   m_lineDecl = (int32) endianInt32( line );


   // the id is not restored, as it is assigned by load order sequence.

   uint32 strid;
   in->read( &strid, sizeof( strid ) );
   m_name = m_module->getString( endianInt32( strid ) );
   if ( m_name == 0 ) {
      return false;
   }

   switch( type_t( type ) ) {
      case tfunc:
         setFunction( new FuncDef( 0 ) );
      return getFuncDef()->load( m_module, in );

      case tclass:
         setClass( new ClassDef( 0, 0 ) );
      return getClassDef()->load( m_module, in );

      case tvar:
         setVar( new VarDef() );
      return getVarDef()->load( m_module, in );

      case tconst:
         setConst( new VarDef() );
      return getVarDef()->load( m_module, in );


      case tprop:
         setProp( new VarDef() );
      return getVarDef()->load( m_module, in );


      case tglobal:
         setGlobal();
      break;

      case tinst:
      {
         in->read(   &strid , sizeof( strid ) );
         strid = endianInt32( strid );
         Symbol *other = m_module->getSymbol( strid );
         if ( other == 0 )
            return false;
         setInstance( other );
      }
      break;

      case tundef:
      case tlocal:
      case tparam:
      case tlocalundef:
      case tattribute:
         m_type = type_t( type );
      break;

      default:
         // we don't expect anything else, included textfunc
         return false;
   }
   return true;
}

bool Symbol::save( Stream *out ) const
{
   byte type = (byte) m_type;
   byte flags = m_flags;
   int32 line = endianInt32( m_lineDecl );
   uint16 pos = endianInt16( m_itemPos );
   out->write( &type, sizeof( type ) );
   out->write( &flags, sizeof( flags ) );
   out->write( &pos, sizeof( pos ) );
   out->write( &line, sizeof( line ) );
   // the ID is not serialized, as it is determined by the module

   uint32 strid = endianInt32( m_name->id() );
   out->write( &strid, sizeof( strid ) );

   switch( m_type ) {
      case tfunc: getFuncDef()->save( out ); break;
      case tclass: getClassDef()->save( out ); break;
      case tvar:
      case tconst:
      case tprop: getVarDef()->save( out ); break;
      case tinst:
         strid = endianInt32( getInstance()->id() );
         out->write( &strid, sizeof( strid ) );
      break;
   }
   return true;
}


FuncDef::FuncDef( uint32 offset ):
   m_offset( offset ),
   m_params( 0 ),
   m_locals( 0 ),
   m_undefined( 0 ),
   m_onceItemId( NO_STATE )
{
}

FuncDef::~FuncDef()
{
}

Symbol *FuncDef::addParameter( Symbol *sym )
{
   sym->itemId( m_params++ );
   sym->setParam();
   m_symtab.add( sym );
   return sym;
}


Symbol *FuncDef::addLocal( Symbol *sym )
{
   sym->itemId( m_locals++ );
   sym->setLocal();
   m_symtab.add( sym );
   return sym;
}


Symbol *FuncDef::addUndefined( Symbol *sym )
{
   sym->itemId( m_undefined++ );
   sym->setLocalUndef();
   m_symtab.add( sym );
   return sym;
}


bool FuncDef::save( Stream *out ) const
{
   uint32 offset = endianInt32(m_offset);
   out->write( &offset, sizeof( offset ) );
   uint16 locs = endianInt16( m_locals );
   uint16 params = endianInt16( m_params );
   out->write( &locs, sizeof( locs ) );
   out->write( &params, sizeof( params ) );

   offset = endianInt32(m_onceItemId);
   out->write( &offset, sizeof( offset ) );

   return m_symtab.save( out );
}


bool FuncDef::load( Module *mod, Stream *in )
{
   uint32 offset;
   in->read( &offset, sizeof( offset ) );
   m_offset = endianInt32( offset );
   uint16 loc;
   in->read( &loc, sizeof( loc ) );
   m_locals = endianInt16( loc );
   in->read( &loc, sizeof( loc ) );
   m_params = endianInt16( loc );

   in->read( &offset, sizeof( offset ) );
   m_onceItemId = endianInt32( offset );

   return m_symtab.load( mod, in );
}

InheritDef::~InheritDef()
{
   ListElement *iter = m_params.begin();
   while( iter != 0 ) {
      VarDef *param = (VarDef *) iter->data();
      delete param;
      iter = iter->next();
   }
}


void InheritDef::addParameter( VarDef *def )
{
   //creates a copy of the passed object.
   m_params.pushBack( def );
}


bool InheritDef::save( Stream *out ) const
{
   uint32 parentId = endianInt32( m_baseClass->id() );
   out->write( &parentId, sizeof( parentId ) );

   uint32 size = endianInt32( m_params.size() );
   out->write( &size, sizeof( size ) );

   ListElement *iter = m_params.begin();
   while( iter != 0 ) {
      const VarDef *param = (const VarDef *) iter->data();
      param->save( out );
      iter = iter->next();
   }

   return true;
}

bool InheritDef::load( Module *mod, Stream *in )
{
   uint32 parentId;
   in->read(  &parentId , sizeof( parentId ) );
   parentId = endianInt32( parentId );
   m_baseClass = mod->getSymbol( parentId );
   if ( m_baseClass == 0 )
      return false;

   uint32 size;
   in->read(  &size , sizeof( size ) );
   size = endianInt32( size );

   for( uint32 i = 0; i < size; i++ )
   {
      VarDef *vd = new VarDef();
      m_params.pushBack( vd ); // avoid leak in case of failure.
      if ( ! vd->load( mod, in ) )
         return false;
   }

   return true;
}


ClassDef::ClassDef( uint32 offset, Symbol *ext_ctor ):
   FuncDef( offset ),
   m_constructor( ext_ctor ),
   m_properties( &traits::t_stringptr, &traits::t_voidp )
{}

ClassDef::~ClassDef()
{
   MapIterator iterp = m_properties.begin();
   while( iterp.hasCurrent() )
   {
      VarDef *vd = *(VarDef **) iterp.currentValue();
      delete vd;
      iterp.next();
   }

   ListElement *iteri = m_inheritance.begin();
   while( iteri != 0 )
   {
      InheritDef *def = (InheritDef *) iteri->data();
      delete def;
      iteri = iteri->next();
   }
}

void ClassDef::addProperty( const String *name, VarDef *definition )
{
   MapIterator iter;
   if ( m_properties.find( name, iter ) ) {
      VarDef **vd =(VarDef **) iter.currentValue();
      delete *vd;
      *vd = definition;
   }
   else
      m_properties.insert( name, definition );
}

VarDef *ClassDef::getProperty( const String *name ) const
{
   VarDef **vd = (VarDef **) m_properties.find( name );
   if ( vd != 0 )
      return *vd;
   return 0;
}


bool ClassDef::addInheritance( InheritDef *parent_class )
{
   Symbol *parent = parent_class->base();
   if ( getProperty( parent->name() ) != 0 )
      return false;

   m_inheritance.pushBack( parent_class );
   addProperty( &parent->name(), new VarDef( VarDef::t_base, parent ) );
   return true;
}


bool ClassDef::inheritsFrom( const String &find_name ) const
{
   ListElement *iter = m_inheritance.begin();

   // perform a flat-first search ? -- not for now
   while( iter != 0 ) {
      const InheritDef *def = (const InheritDef *) iter->data();
      const Symbol *i = def->base();
      if( i->fromClass( find_name ) )
         return true;
      iter = iter->next();
   }
   return false;
}

bool ClassDef::save( Stream *out ) const
{
   if ( ! FuncDef::save( out ) )
      return false;

   uint32 has;

   // Writing of the has clause
   //TODO: optimize count of has/hasnt clause.
   has = endianInt32( m_has.size() );
   out->write(   &has , sizeof( has ) );

   has = endianInt32( m_hasnt.size() );
   out->write(   &has , sizeof( has ) );

   // writing has clause
   ListElement *sym_iter = m_has.begin();
   while( sym_iter != 0 ) {
      const Symbol *sym = (const Symbol *) sym_iter->data();
      has = endianInt32( sym->id() );
      out->write( &has , sizeof( has ) );
      sym_iter = sym_iter->next();
   }

   // writing hasn't clause
   sym_iter = m_hasnt.begin();
   while( sym_iter != 0 ) {
      const Symbol *sym = (const Symbol *) sym_iter->data();
      has = endianInt32( sym->id() );
      out->write( &has , sizeof( has ) );
      sym_iter = sym_iter->next();
   }

   // Have we got a constructor?
   if( m_constructor == 0 ) {
      has = 0xffFFffFF;
   }
   else {
      has =  endianInt32( (uint32)m_constructor->id() );
   }
   out->write( &has , sizeof( has ) );

   // now save the property table
   has = endianInt32(m_properties.size());
   out->write( &has , sizeof( has ) );

   MapIterator iter = m_properties.begin();
   while( iter.hasCurrent() )
   {
      const VarDef *value = *(const VarDef **) iter.currentValue();
      const String *key = *(const String **) iter.currentKey();
      uint32 id = endianInt32( key->id() );
      out->write( &id , sizeof( id ) );
      if ( ! value->save( out ) )
         return false;
      iter.next();
   }

   // and finally the inheritance list
   has = endianInt32(m_inheritance.size());
   out->write(   &has , sizeof( has ) );
   ListElement *iiter = m_inheritance.begin();
   while( iiter != 0 )
   {
      const InheritDef *def = (const InheritDef *) iiter->data();
      if ( ! def->save( out ) )
         return false;
      iiter = iiter->next();
   }

   return true;
}

bool ClassDef::load( Module *mod, Stream *in )
{
   if ( ! FuncDef::load( mod, in ) )
      return false;

   uint32 value, has_count, hasnt_count;
   m_has.clear();
   m_hasnt.clear();

   in->read( &value , sizeof( value ) );
   has_count = endianInt32(value);

   in->read( &value , sizeof( value ) );
   hasnt_count = endianInt32(value);

   // reading has clause
   while( has_count > 0 ) {
      uint32 has;
      in->read( &has , sizeof( has ) );
      m_has.pushBack( mod->getSymbol( endianInt32( has ) ) );
      --has_count;
   }

   // reading hasnt clause
   while( hasnt_count > 0 ) {
      uint32 hasnt;
      in->read( &hasnt , sizeof( hasnt ) );
      m_hasnt.pushBack( mod->getSymbol( endianInt32( hasnt ) ) );
      --hasnt_count;
   }

   in->read(   &value , sizeof( value ) );
   value = endianInt32( value );
   // Have we got a constructor?
   if( value == 0xffFFffFF ) {
      m_constructor = 0 ;
   }
   else {
      m_constructor = mod->getSymbol( value );
      if ( m_constructor == 0 )
         return false;
   }

   // now load the property table
   in->read(   &value , sizeof( value ) );
   value = endianInt32( value );
   for( uint32 i = 0; i < value; i ++ )
   {
      uint32 id;
      in->read(   &id , sizeof( id ) );
      const String *name = mod->getString( endianInt32( id ) );
      if ( name == 0 )
         return false;

      VarDef *def = new VarDef();
      // avoid memleak by early insertion
      m_properties.insert( name, def );

      if ( ! def->load( mod, in ) )
         return false;
   }

   // now load the inheritance table
   in->read(   &value , sizeof( value ) );
   value = endianInt32( value );
   for( uint32 j = 0; j < value; j ++ )
   {
      InheritDef *def = new InheritDef();
      m_inheritance.pushBack( def );
      if ( ! def->load( mod, in ) )
         return false;
   }

   return true;
}

bool VarDef::save( Stream *out ) const
{
   int32 type = endianInt32((int32) m_val_type);
   out->write( &type , sizeof( type ) );

   switch( m_val_type )
   {
      case t_bool:
      {
         int32 val = m_value.val_bool ? 1: 0;
         out->write( &val , sizeof( val ) );
      }
      break;

      case t_int:
      case t_reflective:
      {
         int64 val = endianInt64( m_value.val_int );
         out->write( &val , sizeof( val ) );
      }
      break;

      case t_num:
      {
         numeric num = endianNum( m_value.val_num );
         out->write(   &num , sizeof( num ) );
      }
      break;

      case t_string:
      {
         uint32 val = endianInt32( asString()->id() );
         out->write(  &val , sizeof( val ) );
      }
      break;

      case t_symbol:
      case t_reference:
      case t_base:
      {
         uint32 val = endianInt32( asSymbol()->id() );
         out->write( &val , sizeof( val ) );
      }
   }
   return true;
}

bool VarDef::load( Module *mod, Stream *in )
{
   setNil();

   int32 type;
   in->read( &type , sizeof( type ) );
   m_val_type = (t_type) endianInt32(type);

   switch( m_val_type )
   {
      case t_bool:
      {
         int32 val;
         in->read( &val , sizeof( val ) );
         m_value.val_bool = val != 0;
      }
      break;

      case t_int:
      case t_reflective:
      {
         int64 val;
         in->read(   &val , sizeof( val ) );
         m_value.val_int = endianInt64( val );
      }
      break;

      case t_num:
      {
         numeric val;
         in->read( &val, sizeof( val ) );
         m_value.val_num = endianNum( val );
      }
      break;

      case t_string:
      {
         int32 val;
         in->read(   &val , sizeof( val ) );
         m_value.val_str = mod->getString(endianInt32( val ));
         if ( m_value.val_str  == 0 )
            return false;
      }
      break;

      case t_symbol:
      case t_reference:
      case t_base:
      {
         int32 val;
         in->read(   &val , sizeof( val ) );
         m_value.val_sym = mod->getSymbol(endianInt32( val ));
         if ( m_value.val_sym == 0 )
            return false;
      }
   }
   return true;
}

}

/* end of symbol.cpp */

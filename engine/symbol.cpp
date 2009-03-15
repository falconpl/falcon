/*
   FALCON - The Falcon Programming Language.
   FILE: symbol.cpp

   Provide non-inlineizable symbol definitions
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-9-11

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
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
      case timportalias: delete m_value.v_importalias; break;
      default:
         break;
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
         setFunction( new FuncDef( 0, 0 ) );
      return getFuncDef()->load( m_module, in );

      case tclass:
         setClass( new ClassDef );
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

      case timportalias:
      {
         in->read( &strid , sizeof( strid ) );
         strid = endianInt32( strid );
         const String *name = m_module->getString( strid );
         in->read( &strid , sizeof( strid ) );
         strid = endianInt32( strid );
         const String *origMod = m_module->getString( strid );
         byte b;
         in->read( &b, 1 );
         setImportAlias( name, origMod, b == 1 );
      }
      break;

      case tundef:
      case tlocal:
      case tparam:
      case tlocalundef:
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

      case timportalias:
         strid = endianInt32( getImportAlias()->name()->id() );
         out->write( &strid, sizeof( strid ) );
         strid = endianInt32( getImportAlias()->origModule()->id() );
         out->write( &strid, sizeof( strid ) );
         {
            byte b = getImportAlias()->isOrigFileName() ? 1 : 0;
            out->write( &b, 1 );
         }
      break;

      default:
         break;
   }
   return true;
}


Symbol* Symbol::addParam( const String &param )
{

   Symbol* tbc = this;
   if ( isClass() )
   {
      tbc = getClassDef()->constructor();
      if ( tbc == 0 )
         return this;
   }

   switch( tbc->m_type ) {
      case tfunc: tbc->getFuncDef()->addParameter(m_module->addSymbol( param )); break;
      case textfunc: tbc->getExtFuncDef()->addParam(m_module->addSymbol( param )); break;

      default:
         return this;
   }

   return this;
}

//=================================================================
//
int32 ExtFuncDef::getParam( const String &name )
{
   if ( m_params == 0 )
      return -1;

   Symbol *sym = m_params->findByName( name );
   if ( sym == 0 )
      return -1;
   return sym->itemId();
}

/** Adds a function parameter with the specified ID.
   Consider using Symbol::addParam() instead (candy grammar).
*/
ExtFuncDef &ExtFuncDef::addParam( Symbol *param, int32 id )
{
   if ( m_params == 0 )
      m_params = new SymbolTable;

   param->setParam();
   param->itemId( id == -1 ? m_params->size() : id );
   m_params->add( param );

   return *this;
}

ExtFuncDef::~ExtFuncDef()
{
   delete m_params;
}

//=================================================================
//
FuncDef::FuncDef( byte *code, uint32 codeSize ):
   m_code( code ),
   m_codeSize( codeSize ),
   m_params( 0 ),
   m_locals( 0 ),
   m_undefined( 0 ),
   m_onceItemId( NO_STATE ),
   m_basePC(0)
{
}

FuncDef::~FuncDef()
{
   memFree( m_code );
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
   uint16 locs = endianInt16( m_locals );
   uint16 params = endianInt16( m_params );
   out->write( &locs, sizeof( locs ) );
   out->write( &params, sizeof( params ) );

   uint32 onceId = endianInt32(m_onceItemId);
   out->write( &onceId, sizeof( onceId ) );

   uint32 basePC = endianInt32(m_basePC);
   out->write( &basePC, sizeof( basePC ) );

   uint32 codeSize = endianInt32(m_codeSize);
   out->write( &codeSize, sizeof( codeSize ) );
   if ( m_codeSize > 0 )
   {
      out->write( m_code, m_codeSize );
   }

   return m_symtab.save( out );
}


bool FuncDef::load( Module *mod, Stream *in )
{
   uint16 loc;
   in->read( &loc, sizeof( loc ) );
   m_locals = endianInt16( loc );
   in->read( &loc, sizeof( loc ) );
   m_params = endianInt16( loc );

   uint32 onceItem = 0;
   in->read( &onceItem, sizeof( onceItem ) );
   m_onceItemId = endianInt32( onceItem );

   uint32 basePC = 0;
   in->read( &basePC, sizeof( basePC ) );
   m_basePC = endianInt32( basePC );

   int32 codeSize = 0;
   in->read( &codeSize, sizeof( codeSize ) );
   m_codeSize = endianInt32( codeSize );
   m_code = 0;

   // it's essential to check for errors now.
   if ( ! in->good() )
      return false;

   if ( m_codeSize > 0 )
   {
      m_code = (byte *) memAlloc( m_codeSize );
      in->read( m_code, m_codeSize );
      // it's essential to check for errors now.
      if ( ! in->good() )
         return false;
   }

   return m_symtab.load( mod, in );
}

InheritDef::~InheritDef()
{
}

bool InheritDef::save( Stream *out ) const
{
   uint32 parentId = endianInt32( m_baseClass->id() );
   out->write( &parentId, sizeof( parentId ) );

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

   return true;
}


//=================================================================
//

ClassDef::ClassDef( ObjectFactory fact ):
   FuncDef( 0, 0 ),
   m_constructor( 0 ),
   m_properties( &traits::t_stringptr(), &traits::t_voidp() ),
   m_factory( fact ),
   m_metaclassFor( -1 ),
   m_bFinal( false )
{}

ClassDef::ClassDef( Symbol *ext_ctor, ObjectFactory fact ):
   FuncDef( 0, 0 ),
   m_constructor( ext_ctor ),
   m_properties( &traits::t_stringptr(), &traits::t_voidp() ),
   m_factory( fact ),
   m_metaclassFor( -1 ),
   m_bFinal( false )
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

   uint32 value;
   
   in->read( &value , sizeof( value ) );
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

//===================================================================
// vardef

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

      default:
         return true;
   }

   return out->good();
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

      default:
         break;
   }
   return true;
}

}

/* end of symbol.cpp */

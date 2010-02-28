/*
   FALCON - The Falcon Programming Language.
   FILE: module.cpp

   Falcon module manager
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2004-08-01

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/module.h>
#include <falcon/memory.h>
#include <falcon/symtab.h>
#include <falcon/types.h>
#include <cstring>
#include <falcon/common.h>
#include <falcon/stream.h>
#include <falcon/string.h>
#include <falcon/traits.h>
#include <falcon/pcodes.h>
#include <falcon/mt.h>
#include <falcon/attribmap.h>

#include <string.h>

namespace Falcon {

const uint32 Module::c_noEntry = 0xFFffFFff;

Module::Module():
   m_refcount(1),
   m_language( "C" ),
   m_modVersion( 0 ),
   m_engineVersion( 0 ),
   m_lineInfo( 0 ),
   m_loader(0),
   m_serviceMap( &traits::t_string(), &traits::t_voidp() ),
   m_attributes(0)
{
   m_strTab = new StringTable;
   m_bOwnStringTable = true;
}


Module::~Module()
{
   // The depend table points to the string table
   // so there is no need to check it.

   // the services in the service table are usually static,
   // ... in case they need to be dynamic, the subclass of
   // ... the module will take care of them.

   delete m_lineInfo;

   delete m_attributes;

   if ( m_bOwnStringTable )
      delete m_strTab;
}

DllLoader &Module::dllLoader()
{
   if ( m_loader == 0 )
      m_loader = new DllLoader;

   return *m_loader;
}

void Module::addDepend( const String &name, const String &module, bool bPrivate, bool bFile )
{
   m_depend.addDependency( name, module, bPrivate, bFile );
}

void Module::addDepend( const String &name, bool bPrivate, bool bFile )
{
   m_depend.addDependency( name, name, bPrivate, bFile );
}


Symbol *Module::addGlobal( const String &name, bool exp )
{
   if ( m_symtab.findByName( name ) != 0 )
      return 0;

   Symbol *sym = new Symbol( this, m_symbols.size(), name, exp );
   sym->setGlobal( );
   m_symbols.push( sym );

   sym->itemId( m_symtab.size() );
   m_symtab.add( sym );
   return sym;
}

void Module::addSymbol( Symbol *sym )
{
   sym->id( m_symbols.size() );
   sym->module( this );
   m_symbols.push( sym );
}

Symbol *Module::addSymbol()
{
   Symbol *sym = new Symbol( this );
   sym->id( m_symbols.size() );
   m_symbols.push( sym );
   return sym;
}

Symbol *Module::addSymbol( const String &name )
{
   Symbol *sym = new Symbol( this, m_symbols.size(), name, false );
   m_symbols.push( sym );
   return sym;
}

Symbol *Module::addGlobalSymbol( Symbol *sym )
{
   if ( sym->id() >= m_symbols.size() || m_symbols.symbolAt( sym->id() ) != sym ) {
      sym->id( m_symbols.size() );
      m_symbols.push( sym );
   }

   sym->itemId( m_symtab.size() );
   sym->module( this );
   m_symtab.add( sym );
   return sym;
}

void Module::adoptStringTable( StringTable *st, bool bOwn )
{
   if ( m_bOwnStringTable )
      delete m_strTab;
   m_strTab = st;
   m_bOwnStringTable = bOwn;
}


Symbol *Module::addConstant( const String &name, int64 value, bool exp )
{
   Symbol *sym = new Symbol( this, name );
   sym->exported( exp );
   sym->setConst( new VarDef( value ) );
   return addGlobalSymbol( sym );
}

Symbol *Module::addConstant( const String &name, numeric value, bool exp )
{
   Symbol *sym = new Symbol( this, name );
   sym->exported( exp );
   sym->setConst( new VarDef( value ) );
   return addGlobalSymbol( sym );
}

Symbol *Module::addConstant( const String &name, const String &value, bool exp )
{
   Symbol *sym = new Symbol( this, name );
   sym->exported( exp );
   sym->setConst( new VarDef( addString( value ) ) );
   return addGlobalSymbol( sym );
}


Symbol *Module::addExtFunc( const String &name, ext_func_t func, void *extra, bool exp )
{
   Symbol *sym = findGlobalSymbol( name );
   if ( sym == 0 )
   {
      sym = addSymbol(name);
      addGlobalSymbol( sym );
   }
   sym->setExtFunc( new ExtFuncDef( func, extra ) );
   sym->exported( exp );
   return sym;
}

Symbol *Module::addFunction( const String &name, byte *code, uint32 size, bool exp )
{
   Symbol *sym = findGlobalSymbol( name );
   if ( sym == 0 )
   {
      sym = addSymbol(name);
      addGlobalSymbol( sym );
   }
   sym->setFunction( new FuncDef( code, size ) );
   sym->exported( exp );
   return sym;
}


Symbol *Module::addClass( const String &name, Symbol *ctor_sym, bool exp )
{
   if ( m_symtab.findByName( name ) != 0 )
      return 0;

   Symbol *sym = new Symbol( this, m_symbols.size(), name, exp );
   sym->setClass( new ClassDef( ctor_sym ) );
   m_symbols.push( sym );

   sym->itemId( m_symtab.size() );
   m_symtab.add( sym );

   return sym;
}

Symbol *Module::addSingleton( const String &name, Symbol *ctor_sym, bool exp )
{
   String clName = "%" + name;

   // symbol or class symbol already present?
   if ( m_symtab.findByName( name ) != 0 || m_symtab.findByName( clName ) != 0 )
      return 0;

   // create the class symbol (never exported)
   Symbol *clSym = addClass( clName, ctor_sym, false );

   // create a singleton instance of the class.
   Symbol *objSym = new Symbol( this, name );
   objSym->setInstance( clSym );
   objSym->exported( exp );
   addGlobalSymbol( objSym );

   return objSym;
}

Symbol *Module::addSingleton( const String &name, ext_func_t ctor, bool exp )
{
   if( ctor != 0 )
   {
      String ctor_name = name + "._init";
      Symbol *sym = addExtFunc( ctor_name, ctor, false );
      if ( sym == 0 )
         return 0;

      return addSingleton( name, sym, exp );
   }
   else
   {
      return addSingleton( name, (Symbol*)0, exp );
   }
}


Symbol *Module::addClass( const String &name, ext_func_t ctor, bool exp )
{
   String ctor_name = name + "._init";
   Symbol *sym = addExtFunc( ctor_name, ctor, false );
   if ( sym == 0 )
      return 0;
   return addClass( name, sym, exp );
}

VarDef& Module::addClassProperty( Symbol *cls, const String &prop )
{
   ClassDef *cd = cls->getClassDef();
   VarDef *vd = new VarDef();
   cd->addProperty( addString( prop ), vd );
   return *vd;
}

VarDef& Module::addClassMethod( Symbol *cls, const String &prop, Symbol *method )
{
   ClassDef *cd = cls->getClassDef();
   VarDef *vd = new VarDef( method );
   cd->addProperty( addString( prop ), vd );
   return *vd;
}

VarDef& Module::addClassMethod( Symbol *cls, const String &prop, ext_func_t method_func )
{
   String name = cls->name() + "." + prop;
   Symbol *method = addExtFunc( name, method_func, false );
   return addClassMethod( cls, prop, method );
}

String *Module::addCString( const char *pos, uint32 size )
{
   String *ret = stringTable().find( pos );
   if ( ret == 0 ) {
      char *mem = (char *)memAlloc(size+1);
      memcpy( mem, pos, size );
      mem[size] = 0;
      ret = new String();
      ret->adopt( mem, size, size + 1 );
      stringTable().add( ret );
   }
   return ret;
}

String *Module::addString( const String &st )
{
   String *ret = stringTable().find( st );
   if ( ret == 0 ) {
      ret = new String( st );
      ret->bufferize();
      ret->exported( st.exported() );
      stringTable().add( ret );
   }
   return ret;
}

String *Module::addString( const String &st, bool exported )
{
   String *ret = stringTable().find( st );
   if ( ret == 0 ) {
      ret = new String( st );
      ret->exported( exported );
      ret->bufferize();
      stringTable().add( ret );
   }
   else {
      ret->exported( exported );
   }

   return ret;
}

uint32 Module::addStringID( const String &st, bool exported )
{
   uint32 ret = stringTable().size();
   String* s = new String( st );
   s->exported( exported );
   s->bufferize();
   stringTable().add( s );

   return ret;
}

bool Module::save( Stream *out, bool skipCode ) const
{
   // save header informations:
   const char *sign = "FM";
   out->write( sign, 2 );

   char ver = pcodeVersion();
   out->write( &ver, 1 );
   ver = pcodeSubVersion();
   out->write( &ver, 1 );

   // serializing module and engine versions
   uint32 iver = endianInt32( m_modVersion );
   out->write( &iver, sizeof( iver ) );
   iver = endianInt32( m_engineVersion );
   out->write( &iver, sizeof( iver ) );

   // serialize the module tables.
   //NOTE: all the module tables are saved 32 bit alinged.

   if ( ! stringTable().save( out ) )
      return false;

   if ( ! m_symbols.save( out ) )
      return false;

   if ( ! m_symtab.save( out ) )
      return false;

   if ( ! m_depend.save( out ) )
      return false;


   if( m_attributes != 0 )
   {
      iver = endianInt32(1);
      out->write( &iver, sizeof( iver ) );

      if ( ! m_attributes->save( this, out ) )
         return false;
   }
   else {
      iver = endianInt32(0);
      out->write( &iver, sizeof( iver ) );
   }

   if ( m_lineInfo != 0 )
   {
      int32 infoInd = endianInt32( 1 );
      out->write( &infoInd, sizeof( infoInd ) );

      if (  ! m_lineInfo->save( out ) )
         return false;
   }
   else {
      int32 infoInd = endianInt32( 0 );
      out->write( &infoInd, sizeof( infoInd ) );
   }

   return out->good();
}

bool Module::load( Stream *is, bool skipHeader )
{
   // verify module version informations.
   if ( !  skipHeader )
   {
      char c;
      is->read( &c, 1 );
      if( c != 'F' )
         return false;
      is->read( &c, 1 );
      if( c != 'M' )
         return false;
      is->read( &c, 1 );
      if ( c != pcodeVersion() )
         return false;
      is->read( &c, 1 );
      if ( c != pcodeSubVersion() )
         return false;
   }

   // Load the module and engine version.
   uint32 ver;
   is->read( &ver, sizeof( ver ) );
   m_modVersion = endianInt32( ver );
   is->read( &ver, sizeof( ver ) );
   m_engineVersion = endianInt32( ver );

   /*TODO: see if there is a localized table around and use that instead.
      If so, skip our internal strtable.
   */
   if ( ! stringTable().load( is ) )
      return false;

   if ( ! m_symbols.load( this, is ) )
      return false;

   if ( ! m_symtab.load( this, is ) )
      return false;

   if ( ! m_depend.load( this, is ) )
      return false;

   is->read( &ver, sizeof( ver ) );
   if( ver != 0 )
   {
      m_attributes = new AttribMap;
      if ( ! m_attributes->load( this, is ) )
         return false;
   }

   // load lineinfo indicator.
   int32 infoInd;
   is->read( &infoInd, sizeof( infoInd ) );
   infoInd = endianInt32( infoInd );
   if( infoInd != 0 )
   {
      m_lineInfo = new LineMap;
      if ( ! m_lineInfo->load( is ) )
         return false;
   }
   else
      m_lineInfo = 0;

   return is->good();
}


uint32 Module::getLineAt( uint32 pc ) const
{
   if ( m_lineInfo == 0 || m_lineInfo->empty() )
      return 0;

   MapIterator iter;
   if( m_lineInfo->find( &pc, iter ) )
   {
      return *(uint32 *) iter.currentValue();
   }
   else {
      iter.prev();
      if( iter.hasCurrent() )
         return *(uint32 *) iter.currentValue();
      return 0;
   }
}

void Module::addLineInfo( uint32 pc, uint32 line )
{
   if ( m_lineInfo == 0 )
      m_lineInfo = new LineMap;

   m_lineInfo->addEntry( pc, line );
}


void Module::setLineInfo( LineMap *infos )
{
   delete m_lineInfo;
   m_lineInfo = infos;
}

Service *Module::getService( const String &name ) const
{
   Service **srv = (Service **) m_serviceMap.find( &name );
   if ( srv != 0 )
   {
      return *srv;
   }
   return 0;
}

bool Module::publishService( Service *sp )
{
   Service **srv = (Service **) m_serviceMap.find( &sp->getServiceName() );
   if ( srv != 0 )
   {
      return false;
   }
   else {
      m_serviceMap.insert( &sp->getServiceName(), sp );
   }
   return true;
}


char Module::pcodeVersion() const
{
   return FALCON_PCODE_VERSION;
}

char Module::pcodeSubVersion() const
{
   return FALCON_PCODE_MINOR;
}

void Module::getModuleVersion( int &major, int &minor, int &revision ) const
{
   major = (m_modVersion ) >> 16;
   minor = (m_modVersion & 0xFF00 ) >> 8;
   revision = m_modVersion & 0xFF;
}

void Module::getEngineVersion( int &major, int &minor, int &revision ) const
{
   major = (m_engineVersion ) >> 16;
   minor = (m_engineVersion & 0xFF00 ) >> 8;
   revision = m_engineVersion & 0xFF;
}

DllLoader *Module::detachLoader()
{
   DllLoader *ret = m_loader;
   m_loader = 0;
   return ret;
}


void Module::incref() const
{
   atomicInc( m_refcount );
}

void Module::decref() const
{
   if( atomicDec( m_refcount ) <= 0 )
   {
      Module *deconst = const_cast<Module *>(this);
      DllLoader *loader = deconst->detachLoader();
      delete deconst;
      delete loader;
   }
}

bool Module::saveTableTemplate( Stream *stream, const String &encoding ) const
{
   stream->writeString( "<?xml version=\"1.0\" encoding=\"" );
   stream->writeString( encoding );
   stream->writeString( "\"?>\n" );
   return stringTable().saveTemplate( stream, name(), language() );
}


String Module::absoluteName( const String &module_name, const String &parent_name )
{
   if ( module_name.getCharAt(0) == '.' )
   {
      // notation .name
      if ( parent_name.size() == 0 )
         return module_name.subString( 1 );
      else {
         // remove last part of parent name
         uint32 posDot = parent_name.rfind( "." );
         // are there no dot? -- we're at root elements
         if ( posDot == String::npos )
            return module_name.subString( 1 );
         else
            return parent_name.subString( 0, posDot ) + module_name; // "." is included.
      }
   }
   else if ( module_name.find( "self." ) == 0 )
   {
      if ( parent_name.size() == 0 )
         return module_name.subString( 5 );
      else
         return parent_name + "." + module_name.subString( 5 );
   }
   else
      return module_name;
}


void Module::addAttribute( const String &name, VarDef* vd )
{
   if( m_attributes == 0 )
      m_attributes = new AttribMap;

   m_attributes->insertAttrib( name, vd );
}


void Module::rollBackSymbols( uint32 nsize )
{
   uint32 size = symbols().size();

   for (uint32 pos =  nsize; pos < size; pos ++ )
   {
      Symbol *sym = symbols().symbolAt( pos );
      symbolTable().remove( sym->name() );
      delete sym;
   }

   symbols().resize(nsize);
}

}
/* end module.cpp */

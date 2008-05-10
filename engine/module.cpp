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
#include <falcon/enginedata.h>

namespace Falcon {

const uint32 Module::c_noEntry = 0xFFffFFff;

Module::Module():
   m_code( 0 ),
   m_codeSize( 0 ),
   m_entry( c_noEntry ),
   m_refcount(1),
   m_lineInfo( 0 ),
   m_modVersion( 0 ),
   m_engineVersion( 0 ),
   m_loader(0),
   m_language( "C" ),
   m_serviceMap( &traits::t_string, &traits::t_voidp )
{}


Module::~Module()
{
   if ( m_code != 0 )
      memFree( m_code );

   // The depend table points to the string table
   // so there is no need to check it.

   // the services in the service table are usually static,
   // ... in case they need to be dynamic, the subclass of
   // ... the module will take care of them.

   delete m_lineInfo;
}

DllLoader &Module::dllLoader()
{
   if ( m_loader == 0 )
      m_loader = new DllLoader;

   return *m_loader;
}

bool Module::addDepend( String *dep )
{
   m_depend.pushBack( dep );
   return true;
}

Symbol *Module::addGlobal( const String &name, bool exp )
{
   if ( m_symtab.findByName( name ) != 0 )
      return 0;

   String *symName = m_strTab.find( name );
   if( symName == 0 ) {
      symName = new String( name );
      m_strTab.add( symName );
   }

   Symbol *sym = new Symbol( this, m_symbols.size(), symName, exp );
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
   String *n = addString( name );
   Symbol *sym = new Symbol( this, m_symbols.size(), n, false );
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

Symbol *Module::addConstant( const String &name, int64 value, bool exp )
{
   Symbol *sym = new Symbol( this, addString( name ) );
   sym->exported( exp );
   sym->setConst( new VarDef( value ) );
   return addGlobalSymbol( sym );
}

Symbol *Module::addConstant( const String &name, numeric value, bool exp )
{
   Symbol *sym = new Symbol( this, addString( name ) );
   sym->exported( exp );
   sym->setConst( new VarDef( value ) );
   return addGlobalSymbol( sym );
}

Symbol *Module::addConstant( const String &name, const String &value, bool exp )
{
   Symbol *sym = new Symbol( this, addString( name ) );
   sym->exported( exp );
   sym->setConst( new VarDef( addString( value ) ) );
   return addGlobalSymbol( sym );
}


Symbol *Module::addExtFunc( const String &name, ext_func_t func, bool exp )
{
   Symbol *sym = addGlobalSymbol( addSymbol(name) );
   sym->setExtFunc( new ExtFuncDef( func ) );
   sym->exported( exp );
   return sym;
}

Symbol *Module::addFunction( const String &name, uint32 offset, bool exp )
{
   Symbol *sym = addGlobalSymbol( addSymbol(name) );
   sym->setFunction( new FuncDef( offset ) );
   sym->exported( exp );
   return sym;
}


Symbol *Module::addClass( const String &name, Symbol *ctor_sym, bool exp )
{
   if ( m_symtab.findByName( name ) != 0 )
      return 0;

   String *symName = m_strTab.find( name );
   if( symName == 0 ) {
      symName = new String( name );
      m_strTab.add( symName );
   }

   Symbol *sym = new Symbol( this, m_symbols.size(), symName, exp );
   sym->setClass( new ClassDef( 0, ctor_sym ) );
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

   // create a singletone instance of the class.
   Symbol *objSym = new Symbol( this, addString( name ) );
   objSym->setInstance( clSym );
   objSym->exported( exp );
   addGlobalSymbol( objSym );

   return objSym;
}

Symbol *Module::addSingleton( const String &name, ext_func_t ctor, bool exp )
{
   String ctor_name = name + "._init";
   Symbol *sym = addExtFunc( ctor_name, ctor, false );
   if ( sym == 0 )
      return 0;

   return addSingleton( name, sym, exp );
}


Symbol *Module::addClass( const String &name, ext_func_t ctor, bool exp )
{
   String ctor_name = name + "._init";
   Symbol *sym = addExtFunc( ctor_name, ctor, false );
   if ( sym == 0 )
      return 0;
   return addClass( name, sym, exp );
}

VarDef *Module::addClassProperty( Symbol *cls, const String &prop )
{
   ClassDef *cd = cls->getClassDef();
   VarDef *vd = new VarDef();
   cd->addProperty( addString( prop ), vd );
   return vd;
}

VarDef *Module::addClassMethod( Symbol *cls, const String &prop, Symbol *method )
{
   ClassDef *cd = cls->getClassDef();
   VarDef *vd = new VarDef( method );
   cd->addProperty( addString( prop ), vd );
   return vd;
}

VarDef *Module::addClassMethod( Symbol *cls, const String &prop, ext_func_t method_func )
{
   String name = cls->name() + "." + prop;
   Symbol *method = addExtFunc( name, method_func, false );
   return addClassMethod( cls, prop, method );
}

String *Module::addCString( const char *pos, uint32 size )
{
   String *ret = m_strTab.find( pos );
   if ( ret == 0 ) {
      char *mem = (char *)memAlloc(size+1);
      memcpy( mem, pos, size );
      mem[size] = 0;
      ret = new String();
      ret->adopt( mem, size, size + 1 );
      m_strTab.add( ret );
   }
   return ret;
}

String *Module::addString( const String &st )
{
   String *ret = m_strTab.find( st );
   if ( ret == 0 ) {
      ret = new String( st );
      ret->bufferize();
      ret->exported( st.exported() );
		m_strTab.add( ret );
   }

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

   if ( ! m_strTab.save( out ) )
      return false;

   if ( ! m_symbols.save( out ) )
      return false;

   if ( ! m_symtab.save( out ) )
      return false;

   if ( ! m_depend.save( out ) )
      return false;

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

   uint32 entry = endianInt32( m_entry );
   out->write( &entry, sizeof( entry ) );

   // save the code
   // later on, the system may rewrite the code size field.
   // this may be useful to use the file that is being created to compile
   // on the fly the bytecode.
   if ( ! skipCode )
   {
      uint32 cs = endianInt32( m_codeSize );
      out->write( &cs, sizeof( cs ) );

      if ( m_codeSize !=  0 )
         out->write( m_code, m_codeSize );
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
   if ( ! m_strTab.load( is ) )
      return false;

   if ( ! m_symbols.load( this, is ) )
      return false;

   if ( ! m_symtab.load( this, is ) )
      return false;

   if ( ! m_depend.load( this, is ) )
      return false;

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

   // Read the entry point
   uint32 cs;
   is->read( &cs, sizeof( cs ) );
   m_entry = endianInt32( cs );

   // read the code
   is->read( &cs, sizeof( cs ) );
   m_codeSize = endianInt32( cs );
   if ( m_codeSize !=  0 ) {
      m_code = (byte *) memAlloc( m_codeSize );
      is->read(  m_code, m_codeSize );
   }

   return is->good();
}

void Module::addMain()
{
   if( code() != 0 && entry() != 0xFFffFFff )
   {
      Symbol *sym = findGlobalSymbol( "__main__" );
      if ( sym == 0 ) {
         sym = addFunction( "__main__", entry() );
         sym->exported(false);
      }
   }
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


void Module::incref()
{
   Engine::atomicInc( m_refcount );
}

void Module::decref()
{
   if( Engine::atomicDec( m_refcount ) <= 0 )
   {
      DllLoader *loader = detachLoader();
      delete this;
      delete loader;
   }
}

bool Module::saveTableTemplate( Stream *stream, const String &encoding ) const
{
   stream->writeString( "<?xml version=\"1.0\" encoding=\"" );
   stream->writeString( encoding );
   stream->writeString( "\"?>\n" );
   return m_strTab.saveTemplate( stream, name(), language() );
}

}
/* end module.cpp */

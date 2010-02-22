/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_srv.cpp

   Configuration parser module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 22 Feb 2010 20:38:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#include <falcon/srv/confparser_srv.h>
#include "confparser_mod.h"

namespace Falcon {

ConfigFileSrv::ConfigFileSrv():
      Service( FALCON_CONFIG_FILE_SRV_NAME ),
      m_pCf(0)
      {}

ConfigFileSrv::~ConfigFileSrv()
{
   delete m_pCf;
}

bool ConfigFileSrv::initialize( const String &fileName, const String &encoding )
{
   m_pCf = new ConfigFile( fileName, encoding );
   return true;
}

void ConfigFileSrv::encoding( const String &encoding ) { m_pCf->encoding( encoding ); }
const String &ConfigFileSrv::encoding() const { return m_pCf->encoding(); }
bool ConfigFileSrv::load() { return m_pCf->load(); }
bool ConfigFileSrv::load( Stream *input ) { return m_pCf->load( input ); }
bool ConfigFileSrv::save() { return m_pCf->save(); }
bool ConfigFileSrv::save( Stream *output ) { return m_pCf->save( output ); }

const String &ConfigFileSrv::errorMessage() const { return m_pCf->errorMessage(); }
long ConfigFileSrv::fsError() const { return m_pCf->fsError(); }
uint32 ConfigFileSrv::errorLine() const { return m_pCf->errorLine(); }

bool ConfigFileSrv::getValue( const String &key, String &value )
{ return m_pCf->getValue( key, value ); }

bool ConfigFileSrv::getValue( const String &section, const String &key, String &value )
{ return m_pCf->getValue( section, key, value ); }

bool ConfigFileSrv::getNextValue( String &value ) { return m_pCf->getNextValue( value ); }

bool ConfigFileSrv::getFirstSection( String &section )
{ return m_pCf->getFirstSection( section ); }

bool ConfigFileSrv::getNextSection( String &nextSection )
{ return m_pCf->getNextValue( nextSection ); }

bool ConfigFileSrv::getFirstKey( const String &prefix, String &key )
{ return m_pCf->getFirstKey( prefix, key ); }

void ConfigFileSrv::addSection( const String &section )
{ m_pCf->addSection( section ); }

bool ConfigFileSrv::getFirstKey( const String &section, const String &prefix, String &key )
{ return m_pCf->getFirstKey( section, prefix, key ); }

bool ConfigFileSrv::getNextKey( String &key )
{ return m_pCf->getNextKey( key ); }

void ConfigFileSrv::setValue( const String &key, String &value )
{ return m_pCf->setValue( key, value ); }

void ConfigFileSrv::setValue( const String &section, const String &key, const String &value )
{ return m_pCf->setValue( section, key, value ); }

void ConfigFileSrv::addValue( const String &key, const String &value )
{ return m_pCf->addValue( key, value ); }

void ConfigFileSrv::addValue( const String &section, const String &key, String value )
{ return m_pCf->addValue( section, key, value ); }

bool ConfigFileSrv::removeValue( const String &key )
{ return m_pCf->removeValue( key ); }

bool ConfigFileSrv::removeValue( const String &section, const String &key )
{ return m_pCf->removeValue( section, key ); }

bool ConfigFileSrv::removeCategory( const String &category )
{ return m_pCf->removeCategory( category ); }

bool ConfigFileSrv::removeCategory( const String &section, const String &category )
{ return m_pCf->removeCategory( section, category ); }

bool ConfigFileSrv::removeSection( const String &key ) { return m_pCf->removeSection( key ); }

void ConfigFileSrv::clearMainSection() { m_pCf->clearMainSection(); }

}

/* end of confparser_srv.cpp */

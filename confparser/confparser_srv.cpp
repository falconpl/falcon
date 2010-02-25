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

ConfigFileService::ConfigFileService():
      Service( CONFIGFILESERVICE_NAME ),
      m_pCf(0)
      {}

ConfigFileService::~ConfigFileService()
{
   delete m_pCf;
}

bool ConfigFileService::initialize( const String &fileName, const String &encoding )
{
   m_pCf = new ConfigFile( fileName, encoding );
   return true;
}

void ConfigFileService::encoding( const String &encoding ) { m_pCf->encoding( encoding ); }
const String &ConfigFileService::encoding() const { return m_pCf->encoding(); }
bool ConfigFileService::load() { return m_pCf->load(); }
bool ConfigFileService::load( Stream *input ) { return m_pCf->load( input ); }
bool ConfigFileService::save() { return m_pCf->save(); }
bool ConfigFileService::save( Stream *output ) { return m_pCf->save( output ); }

const String &ConfigFileService::errorMessage() const { return m_pCf->errorMessage(); }
long ConfigFileService::fsError() const { return m_pCf->fsError(); }
uint32 ConfigFileService::errorLine() const { return m_pCf->errorLine(); }

bool ConfigFileService::getValue( const String &key, String &value )
{ return m_pCf->getValue( key, value ); }

bool ConfigFileService::getValue( const String &section, const String &key, String &value )
{ return m_pCf->getValue( section, key, value ); }

bool ConfigFileService::getNextValue( String &value ) { return m_pCf->getNextValue( value ); }

bool ConfigFileService::getFirstSection( String &section )
{ return m_pCf->getFirstSection( section ); }

bool ConfigFileService::getNextSection( String &nextSection )
{ return m_pCf->getNextValue( nextSection ); }

bool ConfigFileService::getFirstKey( const String &prefix, String &key )
{ return m_pCf->getFirstKey( prefix, key ); }

void ConfigFileService::addSection( const String &section )
{ m_pCf->addSection( section ); }

bool ConfigFileService::getFirstKey( const String &section, const String &prefix, String &key )
{ return m_pCf->getFirstKey( section, prefix, key ); }

bool ConfigFileService::getNextKey( String &key )
{ return m_pCf->getNextKey( key ); }

void ConfigFileService::setValue( const String &key, String &value )
{ return m_pCf->setValue( key, value ); }

void ConfigFileService::setValue( const String &section, const String &key, const String &value )
{ return m_pCf->setValue( section, key, value ); }

void ConfigFileService::addValue( const String &key, const String &value )
{ return m_pCf->addValue( key, value ); }

void ConfigFileService::addValue( const String &section, const String &key, String value )
{ return m_pCf->addValue( section, key, value ); }

bool ConfigFileService::removeValue( const String &key )
{ return m_pCf->removeValue( key ); }

bool ConfigFileService::removeValue( const String &section, const String &key )
{ return m_pCf->removeValue( section, key ); }

bool ConfigFileService::removeCategory( const String &category )
{ return m_pCf->removeCategory( category ); }

bool ConfigFileService::removeCategory( const String &section, const String &category )
{ return m_pCf->removeCategory( section, category ); }

bool ConfigFileService::removeSection( const String &key ) { return m_pCf->removeSection( key ); }

void ConfigFileService::clearMainSection() { m_pCf->clearMainSection(); }

}

/* end of confparser_srv.cpp */

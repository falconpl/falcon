/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_srv.h

   Configuration parser module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 22 Feb 2010 20:38:39 +0100

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_confparser_srv_H
#define flc_confparser_srv_H

#include <falcon/service.h>
#include <falcon/string.h>

#define CONFIGFILESERVICE_NAME "ConfigFile"

namespace Falcon {

class ConfigFile;

class ConfigFileService: public Service
{
public:
   ConfigFileService();
   virtual ~ConfigFileService();

   virtual bool initialize( const String &fileName, const String &encoding );
   virtual void encoding( const String &encoding );
   virtual const String &encoding() const;

   virtual bool load();
   virtual bool load( Stream *input );
   virtual bool save();
   virtual bool save( Stream *output );

   virtual const String &errorMessage() const;
   virtual long fsError() const;
   virtual uint32 errorLine() const;

   virtual bool getValue( const String &key, String &value ) ;
   virtual bool getValue( const String &section, const String &key, String &value );

   virtual bool getNextValue( String &value );

   virtual bool getFirstSection( String &section );
   virtual bool getNextSection( String &nextSection );

   virtual bool getFirstKey( const String &prefix, String &key );

   /** Adds an empty section (at the bottom of the file).
      \return the newly created section, or 0 if the section is already declared.
   */
   virtual void addSection( const String &section );

   virtual bool getFirstKey( const String &section, const String &prefix, String &key );
   virtual bool getNextKey( String &key );

   virtual void setValue( const String &key, String &value ) ;
   virtual void setValue( const String &section, const String &key, const String &value );

   virtual void addValue( const String &key, const String &value );
   virtual void addValue( const String &section, const String &key, String value );

   virtual bool removeValue( const String &key );
   virtual bool removeValue( const String &section, const String &key );

   virtual bool removeCategory( const String &category );
   virtual bool removeCategory( const String &section, const String &category );

   virtual bool removeSection( const String &key );
   virtual void clearMainSection();

private:
   class ConfigFile *m_pCf;
};

}

#endif

/* end of confparser_srv.h */

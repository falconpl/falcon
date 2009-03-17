/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_mod.h

   Configuration parser module -- module service classes
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: sab nov 25 2006

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
    Configuration parser module -- module service classes
*/

#ifndef flc_confparser_mod_H
#define flc_confparser_mod_H

#include <falcon/dir_sys.h>
#include <falcon/stream.h>
#include <falcon/destroyable.h>
#include <falcon/genericlist.h>
#include <falcon/genericmap.h>

namespace Falcon {

/** This class holds a configuration file line;
   Lines can be empty, they can contain a comment, a section indicator
   and a pair of keys/values (with eventually an extra comment).

   Having the list of lines including non-meaningful lines allows to modify
   the configuration file maintaining the original layout.
*/
class ConfigFileLine:public BaseAlloc
{

public:

   typedef enum {
      t_empty,
      t_section,
      t_keyval,
      t_comment
   } e_type;

   e_type m_type;

   /** Original line.
      If the line is modified or created, this is destroyed.
   */
   String *m_original;

   /** This string contains the key or the section declaration */
   String *m_key;
   /** This string contains the value in t_keyval lines */
   String *m_value;
   /** This string contains the comment that can be attached to any line */
   String *m_comment;

   ConfigFileLine( e_type t, String *original, String *key = 0, String *value = 0, String *comment = 0 );
   ConfigFileLine( String *original );
   ~ConfigFileLine();

   bool parseLine();
};

void deletor_ConfigFileLine( void *memory );

/** Class containing objects needed by the configuration parser.
   This class holds a pointer to the stream used by the config parser,
   the last time it has been read and a cached pointer to the dir service.
*/

class ConfigEntry: public BaseAlloc
{
public:
   /** Single entry, complete from root, i.e.
      a.b.c = z  -> "a.b.c"
   */
   String m_entry;

   /** List of pointers to the ConfigFileLine entries.
      Actually, the entry points to the list ELEMENT containing the line,
      so it is possible to remove it from the file line list.
   */
   List m_values;

};

class ConfigEntryPtrTraits: public ElementTraits
{
public:
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};

class ConfigSection: public Destroyable
{
public:
   String m_name;
   Map m_entries;
   
   virtual ~ConfigSection() {}

   /** List element pointing to the file line where the section is declared.
      Used to remove the whole section
      Zero if undefined/beginning of the file.
   */
   ListElement *m_sectDecl;

   /** List element pointing to the file lines where to put new entries
   If zero, it means add to top of the file
   */
   ListElement *m_additionPoint;

   ConfigSection( const String &name, ListElement *sectLine=0, ListElement *addPoint = 0 );
};

class ConfigSectionPtrTraits: public ElementTraits
{
public:
	virtual uint32 memSize() const;
	virtual void init( void *itemZone ) const;
	virtual void copy( void *targetZone, const void *sourceZone ) const;
	virtual int compare( const void *first, const void *second ) const;
	virtual void destroy( void *item ) const;
   virtual bool owning() const;
};


class ConfigFile: public FalconData
{
   String m_fileName;

   /** List of phisical lines in the file lines */
   List m_lines;

   /** List of the keys in the main section
      string* -> ConfigFileEntry
   */
   ConfigSection m_rootEntry;

   /** Map of sections (excluding root) if present
      string -> Map (of string*->Sections)
   */
   Map m_sections;
   MapIterator m_sectionIter;

   MapIterator m_keysIter;
   String m_keyMask;

   String m_errorMsg;
   /** FileSystem error.
      In case of FS errors while reading the file, this member is set.
   */
   long m_fsError;
   String m_encoding;

   ListElement *m_currentValue;

   uint32 m_errorLine;
   bool m_bUseUnixComments;
   bool m_bUseUnixSpecs;

   bool getFirstKey_internal( ConfigSection *sect, const String &prefix, String &key );
   void setValue_internal( ConfigSection *sect, const String &key, const String &value );
   void addValue_internal( ConfigSection *sect, const String &key, const String &value );
   bool removeValue_internal( ConfigSection *sect, const String &key );
   bool removeCategory_internal( ConfigSection *sect, const String &key );

public:
   ConfigFile( const String &fileName, const String &encoding );
   virtual ~ConfigFile();

   void encoding( const String &encoding ) { m_encoding = encoding; }
   const String &encoding() const { return m_encoding; }

   bool load();
   bool load( Stream *input );
   bool save();
   bool save( Stream *output );

   const String &errorMessage() const { return m_errorMsg; }
   long fsError() const { return m_fsError; }
   uint32 errorLine() const { return m_errorLine; }

   bool getValue( const String &key, String &value ) ;
   bool getValue( const String &section, const String &key, String &value );

   bool getNextValue( String &value );

   bool getFirstSection( String &section );
   bool getNextSection( String &nextSection );

   bool getFirstKey( const String &prefix, String &key ) {
      return getFirstKey_internal( &m_rootEntry, prefix, key );
   }

   /** Adds an empty section (at the bottom of the file).
      \return the newly created section, or 0 if the section is already declared.
   */
   ConfigSection *addSection( const String &section );

   bool getFirstKey( const String &section, const String &prefix, String &key );
   bool getNextKey( String &key );

   void setValue( const String &key, String &value ) ;
   void setValue( const String &section, const String &key, const String &value );

   void addValue( const String &key, const String &value );
   void addValue( const String &section, const String &key, String value );

   bool removeValue( const String &key );
   bool removeValue( const String &section, const String &key );

   bool removeCategory( const String &category );
   bool removeCategory( const String &section, const String &category );

   bool removeSection( const String &key );
   void clearMainSection();

   virtual void gcMark( uint32 mk ) {}
   virtual FalconData *clone() const { return 0; }
};

}

#endif

/* end of confparser_mod.h */

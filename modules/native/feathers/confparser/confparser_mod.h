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

#ifndef FALCON_CONFPARSER_MOD
#define FALCON_CONFPARSER_MOD

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/mt.h>

namespace Falcon {

class TextReader;
class TextWriter;
class ConfigFile;

/** This class holds a configuration file line;
   Lines can be empty, they can contain a comment, a section indicator
   and a pair of keys/values (with eventually an extra comment).

   Having the list of lines including non-meaningful lines allows to modify
   the configuration file maintaining the original layout.
*/
class ConfigFileLine
{
public:

   typedef enum {
      t_empty,
      t_section,
      t_keyval,
      t_comment
   } e_type;


   ConfigFileLine( e_type t, const String& key, const String& value, const String& comment );
   ConfigFileLine( const String& original );
   ~ConfigFileLine();

   bool parseLine();

   void comment( const String& comment );
   const String& comment() const { return m_comment; }

   /** Sets the key for this entry.
       This changes the line type to a key-value pair.
    */
   void key( const String& k );
   const String& key() const { return m_key; }

   /** Sets the value for this entry.
       This changes the line type to a key-value pair.
    */
   void value( const String& v );
   const String& value() const { return m_value; }

   /** Sets the key and value for this entry.
       This changes the line type to a key-value pair.
    */
   void setKeyValue( const String& k, const String& v );

   /** Sets this entry as a section header.
       This changes the line type to a section header type.
    */
   void setSection( const String& name );

   /** Original line.
      If this line is synthesized or created, this value is empty.
   */
   const String& original() const { return m_original; }

   /** Get the original line or synthetize a value.

      If the line is modified or created, this value is
      synthesized out of the key/value/comment triplet, with this format:

      @code
         key=value<tab>; comment
      @endcode
    */
   void compose( String& target, char_t commentChar = '#' );

private:
   class Private;
   Private* _p;


   e_type m_type;

   String m_original;

   /** This string contains the key or the section declaration */
   String m_key;
   /** This string contains the value in t_keyval lines */
   String m_value;
   String m_comment;

   friend class ConfigSection;
   friend class ConfigFile;
};


class ConfigSection
{
public:
   String m_name;

   ConfigSection( const String &name);
   ConfigSection( ConfigFileLine* firstLine );
   virtual ~ConfigSection();

   bool addLine( ConfigFileLine* line );

   /** Gets the first value associated with a key or a category.
    * \param key A key, or a category (if it ends with '.').
    * \param value the value associated with the found key.
    * \return True if found, false if not found.
    */
   bool getValue( const String &key, String &value );


   /** Gets the more values associated with a key or a category.
    * \param key A key, or a category (if it ends with '.').
    * \param value the value associated with the found key.
    * \return True if found, false if not found.
    *
    * Has undefined behavior if not called after a successful getValue()
    */
   bool getNextValue( const String &key, String& value );
   void setValue( const String &key, String &value ) ;
   void addValue( const String &key, String &value ) ;

   bool removeValue( const String &key );
   bool removeValue( const String &key, const String& value );

   bool removeCategory( const String &category );

   class KeyEnumerator {
   public:
      virtual ~KeyEnumerator(){}
      virtual void operator() (const String& key, const String& value) = 0;
   };

   void enumerateKeys( KeyEnumerator& es ) const;
   void enumerateCategory( KeyEnumerator& es, const String& category, bool trimCategory ) const;

   void clear();
private:
   class Private;
   Private* _p;
   friend class ConfigFile;
   friend class ConfigFileLine;
};


class ConfigFile
{
public:
   ConfigFile();
   virtual ~ConfigFile();

   bool load( TextReader *input );
   bool save( TextWriter *output ) const;

   const String &errorMessage() const { return m_errorMsg; }
   uint32 errorLine() const { return m_errorLine; }

   ConfigSection* mainSection() const;
   ConfigSection* getSection( const String& name ) const;
   ConfigSection *addSection( const String &section );
   bool addSection( ConfigSection* section );
   bool removeSection( const String& name );

   void gcMark( uint32 mk ) { m_mark = mk; }
   uint32 currentMark() const { return m_mark; }

   void lock() const { m_mtx.lock(); }
   void unlock() const { m_mtx.unlock(); }

   class locker {
   public:
      locker( const ConfigFile& file ): m_file(file) { file.lock(); }
      ~locker() { m_file.unlock(); }
   private:
      const ConfigFile& m_file;
   };

   class SectionEnumerator {
   public:
      virtual ~SectionEnumerator(){}
      virtual void operator() (const ConfigSection* section) = 0;
   };

   void enumerateSections( SectionEnumerator& es ) const;
    
   

private:
   String m_fileName;
   String m_keyMask;

   String m_errorMsg;
   uint32 m_errorLine;
   uint32 m_mark;

   bool m_bUseUnixComments;
   bool m_bUseUnixSpecs;
    
   mutable Mutex m_mtx;

   class Private;
   Private* _p;
};

}

#endif

/* end of confparser_mod.h */

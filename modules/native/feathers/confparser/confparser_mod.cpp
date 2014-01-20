/*
   FALCON - The Falcon Programming Language
   FILE: confparser_mod.cpp

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

#include "confparser_mod.h"
#include <falcon/textreader.h>
#include <falcon/textwriter.h>
#include <falcon/mt.h>

#include <falcon/stdstreams.h>

#include <map>
#include <list>

namespace Falcon {


class ConfigSection::Private
{
public:
   typedef std::multimap<String,ConfigFileLine*> EntryMap;
   EntryMap m_entries;

   typedef std::list<ConfigFileLine*> LineList;
   LineList m_lines;

   EntryMap::const_iterator m_searchIterator;

   Private() {}
   ~Private() {
      LineList::iterator iter = m_lines.begin();
      while( iter != m_lines.end() )
      {
         ConfigFileLine* line = *iter;
         delete line;
         ++iter;
      }
   }
};


class ConfigFile::Private
{
public:
   typedef std::list<ConfigSection*> SectionList;
   SectionList m_sections;

   typedef std::map<String,ConfigSection*> SectionMap;
   SectionMap m_sectByName;

   /** List of the keys in the main section
      string* -> ConfigFileEntry
   */
   ConfigSection* m_rootEntry;

   Private() {
      m_rootEntry = new ConfigSection("");
      m_sections.push_back( m_rootEntry );
      m_sectByName[""] = m_rootEntry;
   }

   ~Private()
   {
      SectionList::iterator iter = m_sections.begin();
      while( iter != m_sections.end() )
      {
         ConfigSection* section = *iter;
         delete section;
         ++iter;
      }
   }
};


class ConfigFileLine::Private
{
public:
   ConfigSection::Private::EntryMap::iterator m_posAsEntry;
   ConfigSection::Private::LineList::iterator m_posAsLine;

   Private() {}
   ~Private() {}
};


ConfigFileLine::ConfigFileLine( e_type t, const String& k, const String& v, const String& c ):
   m_type( t ),
   m_key( k ),
   m_value( v ),
   m_comment( c )
{
   _p = new Private;
}


ConfigFileLine::ConfigFileLine( const String& original ):
   m_type( t_empty ),
   m_original( original )
{
   _p = new Private;
   parseLine();
}

ConfigFileLine::~ConfigFileLine()
{
   delete _p;
}

bool ConfigFileLine::parseLine()
{
   m_type = t_empty;
   String tempString;

   enum  {
      comment,
      normal,
      section,
      key,
      postkey,
      prevalue,
      value,
      stringvalue,
      stringescape,
      stringHex,
      stringOctal,
      postvalue
   } state;

   state = normal;
   uint32 trimValuePos = 0;

   for ( uint32 pos = 0; pos < m_original.length(); pos ++ )
   {
      uint32 chr = m_original.getCharAt( pos );
      switch( state ) {
         case normal: // normal state
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original.length() - 1 )
                  m_comment = m_original.subString( pos + 1 );
               m_type = t_comment;
               return true;
            }
            else if ( chr == '=' )
               return false;
            else if ( chr == '\"' )
               return false;
            else if ( chr == '[' )
            {
               m_key.size(0);
               m_type = t_section;
               state = section; // section name
            }
            else if ( chr != ' ' && chr != '\t' )
            {
               m_type = t_keyval;
               m_key.size(0);
               m_key.append( chr );
               state = key; // read key
            }
            // else stay in this state;
         break;

         case section: // read section name
            if ( chr != ']' )
               m_key.append( chr );
            else
               state = postvalue; // post value
         break;

         case key: // read key
            if ( chr == '"' )
               return false;
            else if ( chr == '=' || chr == ':' )
               state = prevalue;
            else if ( chr == ' ' || chr == '\t' )
               state = postkey;
            else
               m_key.append( chr );
         break;

         case postkey:
            if ( chr == '=' || chr == ':' )
               state = prevalue;
            else if ( chr != ' ' && chr != '\t' )
               return false;
         break;

         case prevalue:
            if ( chr == '"' )
            {
               state = stringvalue;
               m_value.size(0);
            }
            else if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original.length() - 1 )
                  m_comment = m_original.subString( pos + 1 );
               return true;
            }
            else if ( chr != ' ' && chr != '\t' )
            {
               state = value;
               m_value.size(0);
               m_value.append( chr );
               trimValuePos = m_value.size();
            }
         break;

         case value:
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original.length() - 1 )
                  m_comment = m_original.subString( pos + 1 );
               m_value.size( trimValuePos );
               return true;
            }

            m_value.append( chr );
            if ( chr != ' ' && chr != '\t' )
               trimValuePos = m_value.size(); // using directly size instead of length
         break;

         case stringvalue:
            if ( chr == '"' )
               state = postvalue;
            else if ( chr == '\\' )
            {
               state = stringescape;
            }
            else
               m_value.append( chr );
         break;

         case stringescape:
            switch( chr )
            {
            case '\\': m_value.append( '\\' ); state = stringvalue; break;
            case '"':  m_value.append( '"' ); state = stringvalue; break;
            case 'n':  m_value.append( '\n' ); state = stringvalue; break;
            case 'b':  m_value.append( '\b' ); state = stringvalue; break;
            case 't':  m_value.append( '\t' ); state = stringvalue; break;
            case 'r':  m_value.append( '\r' ); state = stringvalue; break;
            case 'x': case 'X': tempString.size(0); state = stringHex; break;
            case '0': tempString.size(0); state = stringOctal; break;
            default: m_value.append( chr ); state = stringvalue; break;
            }
         break;

         case stringHex:
            if (  (chr >= '0' && chr <= '9') ||
                     (chr >= 'a' && chr <= 'f') ||
                     (chr >= 'A' && chr <= 'F')
               )
            {
                  tempString.append( chr );
            }
            else
            {
               uint64 retval;
               if ( ! tempString.parseHex( retval ) || retval > 0xFFFFFFFF )
                  return false;
               m_value.append( (uint32) retval );
               m_value.append( chr );
               state = stringvalue;
            }
         break;

         case stringOctal:
            if (  (chr >= '0' && chr <= '7') )
            {
                  tempString.append( chr );
            }
            else
            {
               uint64 retval;
               if ( ! tempString.parseOctal( retval ) || retval > 0xFFFFFFFF )
                  return false;
               m_value.append( (uint32) retval );
               m_value.append( chr );
               state = stringvalue;
            }
         break;

         case postvalue:
            // in postvalue state we wait for a comment.
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original.length() - 1 )
                  m_comment = m_original.subString( pos + 1 );
               return true;
            }
         break;
            
         default:
            break;
      }
   }

   // cleanup
   if ( state == value )
   {
      // trim the value
      m_value.size( trimValuePos );
      return true;
   }
   else if ( state == normal )
   {
      // empty line
      return true;
   }
   else if ( state == postvalue || state == prevalue )
   {
      // we are in a coherent status. If we're in prevalue, the key was left without a value, which is ok
      // i.e.  key = <nothing>

      return true;
   }

   // any other state means we failed parsing
   return false;
}


void ConfigFileLine::comment( const String& c )
{
   m_comment = c;
   m_original.size(0);
}

void ConfigFileLine::key( const String& k )
{
   m_type = t_keyval;
   m_key = k;
   m_original.size(0);
}

void ConfigFileLine::value( const String& v )
{
   m_type = t_keyval;
   m_value = v;
   m_original.size(0);
}

void ConfigFileLine::setKeyValue( const String& k, const String& v )
{
   m_type = t_keyval;
   m_key = k;
   m_value = v;
   m_original.size(0);
}

void ConfigFileLine::setSection( const String& name )
{
   m_type = t_section;
   m_key = name;
   m_value.size(0);
   m_original.size(0);
}


void ConfigFileLine::compose( String& target, char_t cmtchr )
{
   // do we have an original string?
   if( ! m_original.empty() )
   {
      // use it.
      target = m_original;
      return;
   }

   // else, synthesize
   switch( m_type )
   {
   case t_empty:
      target.size(0);
      break;

   case t_comment: case t_keyval:
      if( m_key.empty() )
      {
         target.append(cmtchr);
         target.append(' ');
         target.append( m_comment );
      }
      else {
         target = m_original = m_key + "=" + m_value;
         if( ! m_comment.empty() )
         {
            target += "\t";
            target.append(cmtchr);
            target += " " + m_comment;
         }
      }
      break;

   case t_section:
      if( m_type == t_section )
      {
         m_original = "[" + m_key + "]";
      }

      if( ! m_comment.empty() )
      {
         target += "\t";
         target.append(cmtchr);
         target += " " + m_comment;
      }
      break;
   }

}

//==============================================================
// ConfigSection and traits
//==============================================================


ConfigSection::ConfigSection( const String &name ):
   m_name(name)
{
   _p = 0;
   clear();
}


ConfigSection::ConfigSection( ConfigFileLine* line ):
   m_name(line->m_key)
{
   _p = new Private;
   _p->m_lines.push_back( line );
}


ConfigSection::~ConfigSection()
{
   delete _p;
}

void ConfigSection::clear()
{
   delete _p;
   _p = new Private;
   _p->m_lines.push_back( new ConfigFileLine("["+m_name+"]") );
}

bool ConfigSection::addLine( ConfigFileLine* line )
{
   _p->m_lines.push_back(line);
   if( line->m_type == ConfigFileLine::t_keyval )
   {
      line->_p->m_posAsLine = _p->m_lines.end();
      line->_p->m_posAsLine--;
      line->_p->m_posAsEntry = _p->m_entries.insert( std::make_pair(line->m_key, line) );
   }

   return true;
}


void ConfigSection::setValue( const String &key, String &value )
{
   removeValue(key);
   addValue( key, value );
}

void ConfigSection::addValue( const String &key, String &value )
{
   ConfigFileLine* line = new ConfigFileLine( ConfigFileLine::t_keyval, key, value, "" );
   _p->m_entries.insert( std::make_pair(line->m_key, line) );
}

bool ConfigSection::removeValue( const String &key )
{
   bool done = false;

   Private::EntryMap::iterator iter = _p->m_entries.find( key );
   while( iter != _p->m_entries.end() && iter->first == key )
   {
      ConfigFileLine* line = iter->second;
      Private::EntryMap::iterator old = iter;
      iter++;

      _p->m_lines.erase( line->_p->m_posAsLine );
      _p->m_entries.erase(old);
      delete line;
      done = true;
   }

   return done;
}

bool ConfigSection::removeValue( const String &key, const String& value )
{
   bool done = false;

   Private::EntryMap::iterator iter = _p->m_entries.find( key );
   while( iter != _p->m_entries.end() && iter->first == key )
   {
      ConfigFileLine* line = iter->second;
      Private::EntryMap::iterator old = iter;
      iter++;

      if( value == line->value() )
      {
         _p->m_lines.erase( line->_p->m_posAsLine );
         _p->m_entries.erase(old);
         delete line;
         done = true;
      }
   }

   return done;
}


bool ConfigSection::removeCategory( const String &cat )
{
   bool done = false;

   Private::EntryMap::iterator iter = _p->m_entries.lower_bound( cat );

   String kat;
   if( cat.endsWith(".") )
   {
      kat = cat;
   }
   else {
      kat = cat + ".";
   }

   while( iter != _p->m_entries.end() && iter->first.startsWith(kat) )
   {
      ConfigFileLine* line = iter->second;
      Private::EntryMap::iterator old = iter;
      iter++;

      _p->m_lines.erase( line->_p->m_posAsLine );
      _p->m_entries.erase(old);
      delete line;
      done = true;
   }

   return done;
}


bool ConfigSection::getValue( const String &key, String &value )
{
   Private::EntryMap::const_iterator pos;

   if( key.endsWith(".") )
      pos = _p->m_entries.lower_bound(key);
   else
      pos = _p->m_entries.find(key);

   if ( pos != _p->m_entries.end() && pos->first.startsWith(key) )
   {
      value = pos->second->value();
      _p->m_searchIterator = pos;
      _p->m_searchIterator++;
      return true;
   }

   _p->m_searchIterator = _p->m_entries.end();
   return false;
}


bool ConfigSection::getNextValue( const String &key, String& value )
{
   if( _p->m_searchIterator != _p->m_entries.end() )
   {
      bool isCat = key.endsWith(".");

      if( (isCat && _p->m_searchIterator->first.startsWith(key))
            || ( !isCat && _p->m_searchIterator->first == key) )
      {
         value = _p->m_searchIterator->second->value();
         _p->m_searchIterator++;
         return true;
      }
   }

   return false;
}

void ConfigSection::enumerateKeys( KeyEnumerator& es ) const
{
   Private::EntryMap::const_iterator iter = _p->m_entries.begin();
   while( _p->m_entries.end() != iter )
   {
      const String& key = iter->first;
      const String& value = iter->second->m_value;
      es( key, value );
      ++iter;
   }
}

void ConfigSection::enumerateCategory( KeyEnumerator& es, const String& category, bool trimCategory ) const
{
   String rcat = category;
   if( ! rcat.endsWith(".") )
   {
      rcat.append('.');
   }

   Private::EntryMap::const_iterator iter = _p->m_entries.lower_bound( rcat );
   while( _p->m_entries.end() != iter )
   {
      const String& key = iter->first;
      const String& value = iter->second->m_value;

      if( ! key.startsWith(rcat) )
      {
         return;
      }

      if( trimCategory )
      {
         es(key.subString(rcat.length()), value);
      }
      else
      {
         es( key, value );
      }

      ++iter;
   }
}
//==============================================================
// ConfigFile
//==============================================================


ConfigFile::ConfigFile():
   m_errorLine( 0 ),
   m_mark(0),
   m_bUseUnixComments( false ),
   m_bUseUnixSpecs( false )
{
   _p = new Private;
}


ConfigFile::~ConfigFile()
{
   delete _p;
}


bool ConfigFile::load( TextReader *input )
{
   uint32 chr;
   String currentLine;
   uint32 count = 1;

   // start putting items in the main section
   ConfigSection *current = _p->m_rootEntry;

   //===========================================
   // main loop
   while( ! input->eof() )
   {
      chr = input->getChar();

      if ( chr == '\n' )
      {
         // nextline
         ConfigFileLine *line = new ConfigFileLine( currentLine );
         currentLine.size(0);

         if ( line->parseLine() )
         {
            if( line->m_type == ConfigFileLine::t_section )
            {
               // change current key storage
               current = new ConfigSection( line );

               if( ! addSection( current ) ) {
                  // the section has been stored even if addSection returns false.
                  m_errorMsg = "Duplicated section at line ";
                  m_errorLine = count;
                  m_errorMsg.writeNumber( (int64) count );
                  return false;
               }
            }
            else
            {
               current->addLine( line );
            }
         }
         else {
            m_errorMsg = "Parse failed at line ";
            m_errorLine = count;
            m_errorMsg.writeNumber( (int64) count );
            return false;
         }
         // the line has been taken by the ConfigFileLine
         count ++;
      }
      else {
         currentLine.append( chr );
      }
   }

   return true;
}


bool ConfigFile::save( TextWriter *output ) const
{
   Private::SectionList::const_iterator iter = _p->m_sections.begin();

   while( iter != _p->m_sections.end() )
   {
      ConfigSection* section = *iter;
      ConfigSection::Private::LineList::const_iterator li = section->_p->m_lines.begin();
      ConfigSection::Private::LineList::const_iterator li_end = section->_p->m_lines.end();

      while( li_end != li )
      {
         ConfigFileLine* line = *li;
         String text;
         line->compose( text );
         output->write( text );
         ++li;
      }

      ++iter;
   }
   return true;
}


ConfigSection* ConfigFile::mainSection() const
{
   return _p->m_rootEntry;
}

ConfigSection* ConfigFile::getSection( const String& name ) const
{
   if( name.empty() )
   {
      return _p->m_rootEntry;
   }

   Private::SectionMap::iterator iter = _p->m_sectByName.find( name );
   if( iter == _p->m_sectByName.end() )
   {
      return 0;
   }

   return iter->second;
}


ConfigSection *ConfigFile::addSection( const String &name )
{
   if( getSection(name) != 0 )
   {
      return 0;
   }

   ConfigSection* section = new ConfigSection(name);
   _p->m_sectByName[section->m_name] = section;
   _p->m_sections.push_back(section);
   return section;
}


bool ConfigFile::addSection( ConfigSection* section )
{
   if( getSection(section->m_name) != 0 )
   {
      return false;
   }

   _p->m_sectByName[section->m_name] = section;
   _p->m_sections.push_back(section);
   return true;
}

bool ConfigFile::removeSection( const String& name )
{
   Private::SectionMap::iterator iter = _p->m_sectByName.find( name );
   if( iter == _p->m_sectByName.end() )
   {
      return false;
   }

   ConfigSection* section = iter->second;
   _p->m_sectByName.erase(iter);

   // remove from the list with a linear scan.
   // not efficient, but, how many sections can you have in an hand-written ini-file?
   Private::SectionList::iterator li = _p->m_sections.begin();
   while( li != _p->m_sections.end() )
   {
      if( *li == section )
      {
         _p->m_sections.erase( li );
         break;
      }
      ++li;
   }

   delete section;
   return true;
}


void ConfigFile::enumerateSections( SectionEnumerator& es ) const
{
   Private::SectionList::const_iterator li = _p->m_sections.begin();
   while( li != _p->m_sections.end() )
   {
      ConfigSection* section = *li;
      es(section);
      ++li;
   }
}

}

/* end of confparser_mod.cpp */

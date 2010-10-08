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
#include <falcon/fstream.h>
#include <falcon/transcoding.h>
#include <falcon/stdstreams.h>

namespace Falcon {

ConfigFileLine::ConfigFileLine( e_type t, String *original, String *key, String *value, String *comment ):
   m_type( t ),
   m_original( original ),
   m_key( key ),
   m_value( value ),
   m_comment( comment )
{

}

ConfigFileLine::ConfigFileLine( String *original ):
   m_type( t_empty ),
   m_original( original ),
   m_key( 0 ),
   m_value( 0 ),
   m_comment( 0 )
{
}

ConfigFileLine::~ConfigFileLine()
{
   delete m_key;
   delete m_value;
   delete m_comment;
   delete m_original;
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

   for ( uint32 pos = 0; pos < m_original->length(); pos ++ )
   {
      uint32 chr = m_original->getCharAt( pos );
      switch( state ) {
         case normal: // normal state
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original->length() - 1 )
                  m_comment = new String( *m_original, pos + 1 );
               m_type = t_comment;
               return true;
            }
            else if ( chr == '=' )
               return false;
            else if ( chr == '\"' )
               return false;
            else if ( chr == '[' )
            {
               m_key = new String;
               m_type = t_section;
               state = section; // section name
            }
            else if ( chr != ' ' && chr != '\t' )
            {
               m_type = t_keyval;
               m_key = new String;
               m_key->append( chr );
               state = key; // read key
            }
            // else stay in this state;
         break;

         case section: // read section name
            if ( chr != ']' )
               m_key->append( chr );
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
               m_key->append( chr );
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
               m_value = new String;
            }
            else if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original->length() - 1 )
                  m_comment = new String( *m_original, pos + 1 );
               return true;
            }
            else if ( chr != ' ' && chr != '\t' )
            {
               state = value;
               m_value = new String;
               m_value->append( chr );
               trimValuePos = m_value->size();
            }
         break;

         case value:
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original->length() - 1 )
                  m_comment = new String( *m_original, pos + 1 );
               m_value->size( trimValuePos );
               return true;
            }

            m_value->append( chr );
            if ( chr != ' ' && chr != '\t' )
               trimValuePos = m_value->size(); // using directly size instead of length
         break;

         case stringvalue:
            if ( chr == '"' )
               state = postvalue;
            else if ( chr == '\\' )
            {
               state = stringescape;
            }
            else
               m_value->append( chr );
         break;

         case stringescape:
            switch( chr )
            {
            case '\\': m_value->append( '\\' ); state = stringvalue; break;
            case '"':  m_value->append( '"' ); state = stringvalue; break;
            case 'n':  m_value->append( '\n' ); state = stringvalue; break;
            case 'b':  m_value->append( '\b' ); state = stringvalue; break;
            case 't':  m_value->append( '\t' ); state = stringvalue; break;
            case 'r':  m_value->append( '\r' ); state = stringvalue; break;
            case 'x': case 'X': tempString.size(0); state = stringHex; break;
            case '0': tempString.size(0); state = stringOctal; break;
            default: m_value->append( chr ); state = stringvalue;
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
               m_value->append( (uint32) retval );
               m_value->append( chr );
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
               m_value->append( (uint32) retval );
               m_value->append( chr );
               state = stringvalue;
            }
         break;

         case postvalue:
            // in postvalue state we wait for a comment.
            if ( chr == ';' || chr == '#' )
            {
               //the rest is a comment
               if ( pos < m_original->length() - 1 )
                  m_comment = new String( *m_original, pos + 1 );
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
      m_value->size( trimValuePos );
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


void deletor_ConfigFileLine( void *memory )
{
   ConfigFileLine *line = (ConfigFileLine *) memory;
   delete line;
}


//=======================================================
// ConfigEntry traits

uint32 ConfigEntryPtrTraits::memSize() const
{
   return sizeof( ConfigEntry * );
}

void  ConfigEntryPtrTraits::init( void *itemZone ) const
{
   ConfigEntry **map = (ConfigEntry **) itemZone;
   *map = 0;
}

void ConfigEntryPtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   ConfigEntry **tgt = (ConfigEntry **) targetZone;
   ConfigEntry *src = (ConfigEntry *) sourceZone;
   *tgt = src;
}

int ConfigEntryPtrTraits::compare( const void *first, const void *second ) const
{
   return -1;
}

void ConfigEntryPtrTraits::destroy( void *item ) const
{
   ConfigEntry **ptr = (ConfigEntry **) item;
   delete *ptr;
}

bool ConfigEntryPtrTraits::owning() const
{
   return true;
}

namespace traits
{
   ConfigEntryPtrTraits &t_ConfigEntryPtr() { static ConfigEntryPtrTraits td; return td; }
}


//==============================================================
// ConfigSection and traits
//==============================================================

ConfigSection::ConfigSection( const String &name, ListElement *begin, ListElement *ae ):
   m_name(name),
   m_entries( &traits::t_stringptr(), &traits::t_ConfigEntryPtr() ),
   m_sectDecl( begin ),
   m_additionPoint( ae )
{
}

uint32 ConfigSectionPtrTraits::memSize() const
{
   return sizeof( ConfigSection *);
}

void ConfigSectionPtrTraits::init( void *itemZone ) const
{
   ConfigSection **sect = (ConfigSection **) itemZone;
   *sect = 0;
}

void ConfigSectionPtrTraits::copy( void *targetZone, const void *sourceZone ) const
{
   ConfigSection **tgt = (ConfigSection **) targetZone;
   ConfigSection *src = (ConfigSection *) sourceZone;
   *tgt = src;
}

int ConfigSectionPtrTraits::compare( const void *first, const void *second ) const
{
   return -1;
}

void ConfigSectionPtrTraits::destroy( void *item ) const
{
   ConfigSection **ptr = (ConfigSection **) item;
   delete *ptr;
}

bool ConfigSectionPtrTraits::owning() const
{
   return true;
}

namespace traits
{
   ConfigSectionPtrTraits &t_ConfigSectionPtr() { static ConfigSectionPtrTraits dt; return dt; }
}

//==============================================================
// ConfigFile
//==============================================================

ConfigFile::ConfigFile( const String &filename, const String &encoding ):
   m_fileName( filename ),
   m_lines( deletor_ConfigFileLine ),
   m_rootEntry( "root", 0 ),
   m_sections( &traits::t_stringptr(), &traits::t_ConfigSectionPtr() ),
   m_fsError(0),
   m_encoding( encoding ),
   m_currentValue( 0 ),
   m_errorLine( 0 ),
   m_bUseUnixComments( false ),
   m_bUseUnixSpecs( false )
{
}

ConfigFile::~ConfigFile()
{}

bool ConfigFile::load()
{
   m_fsError = 0;
   m_errorMsg = "";

   Stream *input = 0;
   //===========================================
   // Initialization

   FileStream stream;
   if ( ! stream.open( m_fileName, BaseFileStream::e_omReadOnly, BaseFileStream::e_smShareRead ) )
   {
      stream.errorDescription( m_errorMsg );
      m_fsError = (long) stream.lastError();
      return false;
   }

   // get a good transcoder
   if ( m_encoding == "" )
      m_encoding = "C";

   input = TranscoderFactory( m_encoding, &stream, false );
   if ( input == 0 )
   {
      m_errorMsg = "Invalid encoding '" + m_encoding + "'";
      return false;
   }
   input = AddSystemEOL( input, true );

   bool ret = load( input );
   delete input;
   stream.close();
   return ret;
}


bool ConfigFile::load( Stream *input )
{
   uint32 chr;
   String *currentLine = 0;
   uint32 count = 1;

   // start putting items in the main section
   ConfigSection *current = &m_rootEntry;

   //===========================================
   // main loop
   while( input->get( chr ) )
   {
      if ( currentLine == 0 )
         currentLine = new String;


      if ( chr == '\n' )
      {
         // nextline
         ConfigFileLine *line = new ConfigFileLine( currentLine );
         if ( line->parseLine() )
         {
            m_lines.pushBack( line );

            if( line->m_type == ConfigFileLine::t_section )
            {
               // change current key storage
               current = new ConfigSection( *line->m_key, m_lines.end(), m_lines.end() );
               m_sections.insert( &current->m_name, current );
            }
            else if( line->m_type == ConfigFileLine::t_keyval )
            {
               ListElement *last = m_lines.end();
               // is the item already present?
               MapIterator pos;
               ConfigEntry *entry;
               if ( current->m_entries.find( line->m_key, pos ) )
               {
                  entry = *(ConfigEntry **) pos.currentValue();
               }
               else {
                  entry = new ConfigEntry;
                  entry->m_entry = *line->m_key;
                  current->m_entries.insert( &entry->m_entry, entry );
               }
               entry->m_values.pushBack( last ); // save this list element
               current->m_additionPoint = m_lines.end();
            }
            else if ( line->m_type == ConfigFileLine::t_keyval )
            {
               // be sure to set insertion point after comments.
               current->m_additionPoint = m_lines.end();
            }
         }
         else {
            m_errorMsg = "Parse failed at line ";
            m_errorLine = count;
            m_errorMsg.writeNumber( (int64) count );
            return false;
         }
         // the line has been taken by the ConfigFileLine
         currentLine = 0;
         count ++;
      }
      else {
         currentLine->append( chr );
      }
   }

   //===========================================
   // cleanup
   if ( input->error() )
      goto error;

   return true;
   //===========================================
   // error handling

error:
   m_fsError = (long) input->lastError();
   input->errorDescription( m_errorMsg );
   return false;
}

bool ConfigFile::save()
{
   Stream *output = 0;
   //===========================================
   // Initialization

   FileStream stream;
   if ( ! stream.create( m_fileName,
      FileStream::e_aUserRead | FileStream::e_aReadOnly,
      FileStream::e_smShareRead ) )
   {
      m_fsError = (long) stream.lastError();
      stream.errorDescription( m_errorMsg );
      return false;
   }

   // get a good transcoder
   if ( m_encoding == "" )
      m_encoding = "C";

   output = TranscoderFactory( m_encoding, &stream, false );
   if ( output == 0 )
   {
      m_errorMsg = "Invalid encoding '" + m_encoding + "'";
      return false;
   }
   output = AddSystemEOL( output, true );

   bool ret = save( output );
   delete output;
   stream.close();
   return ret;
}

bool ConfigFile::save( Stream *output )
{
   ListElement *element = m_lines.begin();
   while( element != 0 && output->good() )
   {
      ConfigFileLine *line = (ConfigFileLine *) element->data();
      if ( line->m_original != 0 )
      {
         output->writeString( *line->m_original );
      }
      else {
         if ( line->m_type == ConfigFileLine::t_keyval )
         {
            output->writeString( *line->m_key );
            if( m_bUseUnixSpecs )
               output->writeString( ":" );
            else
               output->writeString( " = " );

            String tempValue;
            line->m_value->escape( tempValue );
            // any escaped char changes the lenght of the string.
            // so, if the escaped changed the string...
            // We need quotes also if the string contains a comment character.
            if ( tempValue.length() != line->m_value->length() ||
                 line->m_value->find( ";" ) != String::npos ||
                 line->m_value->find( "#" ) != String::npos
                 )
               tempValue = "\"" + tempValue + "\"";   // we need some ""
            output->writeString( tempValue );
         }
         else if ( line->m_type == ConfigFileLine::t_section )
         {
            output->writeString( "[" );
            output->writeString( *line->m_key );
            output->writeString( "]" );
         }

         // if it was an original comment, we would have just re-written in the
         // previous if.
         if ( line->m_comment != 0 )
         {
            if ( m_bUseUnixComments )
               output->writeString( "\t# " );
            else
               output->writeString( "\t; " );

            output->writeString( *line->m_comment );
         }
      }

      output->writeString( "\n" );
      element = element->next();
   }

   if ( !output->good() )
   {
      m_fsError = (long) output->lastError();
      output->errorDescription( m_errorMsg );
      return false;
   }

   return true;
}

bool ConfigFile::getValue( const String &key, String &value )
{
   MapIterator pos;
   if ( ! m_rootEntry.m_entries.find( &key, pos ) )
      return false;

   ConfigEntry *ce = *(ConfigEntry **) pos.currentValue();
   // values of entries is a list of ListElements; each ListElement is actually stored in the
   // file lines list.
   ListElement *le = (ListElement *) ce->m_values.begin()->data();
   ConfigFileLine *line = (ConfigFileLine *) le->data();
   value = (line->m_value) ? *(line->m_value) : "" ;
   m_currentValue = ce->m_values.begin()->next();

   return true;
}

bool ConfigFile::getValue( const String &section, const String &key, String &value )
{
   MapIterator pos;
   if ( ! m_sections.find( &section, pos ) )
      return false;

   ConfigSection *sect = *(ConfigSection **) pos.currentValue();
   if ( ! sect->m_entries.find( &key, pos ) )
      return false;

   ConfigEntry *ce = *(ConfigEntry **) pos.currentValue();
   // values of entries is a list of ListElements; each ListElement is actually stored in the
   // file lines list.
   ListElement *le = (ListElement *) ce->m_values.begin()->data();
   ConfigFileLine *line = (ConfigFileLine *) le->data();
   value = *line->m_value;
   m_currentValue = ce->m_values.begin()->next();

   return true;
}

bool ConfigFile::getNextValue( String &value )
{
   if ( m_currentValue == 0 )
      return false;

   ListElement *le = (ListElement *) m_currentValue->data();
   ConfigFileLine *line = (ConfigFileLine *) le->data();
   value = *line->m_value;
   m_currentValue = m_currentValue->next();
   return true;
}


bool ConfigFile::getFirstKey_internal( ConfigSection *sect, const String &prefix, String &key )
{
   Map *map = &sect->m_entries;

   if ( map->empty() )
      return false;

   MapIterator pos;

   if ( prefix != "" )
   {
      String catPrefix = prefix + ".";
      map->find( &catPrefix, pos );
      if ( ! pos.hasCurrent() )
         return false;

      String *currentKey = *(String **) pos.currentKey();
      if ( currentKey->find( catPrefix ) == 0 )
      {
         m_keysIter = pos;
         m_keyMask = catPrefix;
         key = *currentKey;
      }
      else
         return false;
   }
   else {
      m_keyMask = "";
      m_keysIter = map->begin();
      key = **(String **) m_keysIter.currentKey();
   }

   m_keysIter.next();
   return true;
}


bool ConfigFile::getFirstKey( const String &section, const String &prefix, String &key )
{
   MapIterator sectIter;
   if ( ! m_sections.find( &section, sectIter ) )
      return false;

   ConfigSection *sect = *(ConfigSection **) sectIter.currentValue();
   return getFirstKey_internal( sect, prefix, key );
}

bool ConfigFile::getNextKey( String &key )
{
   if( ! m_keysIter.hasCurrent() )
      return false;

   String *currentKey = *(String **) m_keysIter.currentKey();
   m_keysIter.next();

   if( m_keyMask == "" || currentKey->find( m_keyMask ) == 0 )
   {
      key = *currentKey;
      return true;
   }
   return false;
}


bool ConfigFile::getFirstSection( String &section )
{
   if( m_sections.empty() )
      return false;

   m_sectionIter = m_sections.begin();
   // sections are string -> map * maps; strings, not string *!
   String *sectName = *(String **) m_sectionIter.currentKey();
   section = *sectName;
   m_sectionIter.next();
   return true;
}

bool ConfigFile::getNextSection( String &nextSection )
{
   if( m_sectionIter.hasCurrent() )
   {
      String *sectName = *(String **) m_sectionIter.currentKey();
      nextSection = *sectName;
      m_sectionIter.next();
      return true;
   }

   return false;
}

void ConfigFile::setValue( const String &key, String &value )
{
   setValue_internal( &m_rootEntry, key, value );
}

void ConfigFile::setValue( const String &section, const String &key, const String &value )
{
   MapIterator sectIter;
   ConfigSection *sect;

   if ( m_sections.find( &section, sectIter ) )
   {
      sect = *( ConfigSection **) sectIter.currentValue();
   }
   else {
      // we must add a new section
      sect = addSection( section );
   }

   setValue_internal( sect, key, value );
}

void ConfigFile::addValue( const String &key, const String &value )
{
   addValue_internal( &m_rootEntry, key, value );
}

void ConfigFile::addValue( const String &section, const String &key, String value )
{
   MapIterator sectIter;
   ConfigSection *sect;

   if ( m_sections.find( &section, sectIter ) )
   {
      sect = *( ConfigSection **) sectIter.currentValue();
   }
   else {
      // we must add a new section
      sect = addSection( section );
   }

   addValue_internal( sect, key, value );
}

bool ConfigFile::removeValue( const String &key )
{
   return removeValue_internal( &m_rootEntry, key );
}

bool ConfigFile::removeValue( const String &section, const String &key )
{
   MapIterator sectIter;
   ConfigSection *sect;

   if ( ! m_sections.find( &section, sectIter ) )
   {
      return false;
   }

   sect = *( ConfigSection **) m_sectionIter.currentValue();
   return removeValue_internal( sect, key );
}


bool ConfigFile::removeCategory_internal( ConfigSection *sect, const String &category )
{
   String key;
   if( ! getFirstKey_internal( sect, category, key ) )
      return false;

   String key1 = key;
   while( getNextKey( key ) )
   {
      removeValue_internal( sect, key1 );
      key1 = key;
   }

   removeValue_internal( sect, key1 );
	return true;
}

bool ConfigFile::removeCategory( const String &cat )
{
   return removeCategory_internal( &m_rootEntry, cat );
}

bool ConfigFile::removeCategory( const String &section, const String &cat )
{
   MapIterator sectIter;
   ConfigSection *sect;

   if ( ! m_sections.find( &cat, sectIter ) )
   {
      return false;
   }

   sect = *( ConfigSection **) m_sectionIter.currentValue();
   return removeCategory_internal( sect, cat );
}


ConfigSection *ConfigFile::addSection( const String &section )
{
   // check if the section already exists:
   MapIterator sectIter;
   if ( m_sections.find( &section, sectIter ) )
   {
      return 0;
   }

   // create the section
   ConfigFileLine *sectLine = new ConfigFileLine(
         ConfigFileLine::t_section, 0, new String( section ) );

   // add a section at the bottom of the file.
   m_lines.pushBack( sectLine );
   ConfigSection *sect = new ConfigSection( section, m_lines.end(), m_lines.end() );
   m_sections.insert( &sect->m_name, sect );
   return sect;
}

void ConfigFile::setValue_internal( ConfigSection *sect, const String &key, const String &value )
{
   MapIterator pos;
   ConfigEntry *entry;

   // we must find the first entry, if present.
   if ( sect->m_entries.find( &key, pos ) )
   {
      entry = *(ConfigEntry **) pos.currentValue();
   }
   else {
      addValue_internal( sect, key, value );
      return;
   }

   // the first value of the entry must be changed; the other must be deletd.
   ListElement *valline = entry->m_values.begin();
   if ( valline == 0 )
   {
      // overkill, should never happen.
      addValue_internal( sect, key, value );
      return;
   }

   // change the first line value
   ListElement *cfline = (ListElement *) valline->data();
   ConfigFileLine *line = (ConfigFileLine *) cfline->data();
   *line->m_value = value;
   // by deleting the original, we signal this line must be recomputed.
   delete line->m_original;
   line->m_original = 0;

   // then remove the other values
   valline = valline->next();
   while( valline != 0 )
   {
      ListElement *toBeDeleted = (ListElement *) valline->data();
      m_lines.erase( toBeDeleted );
      valline = entry->m_values.erase( valline );
   }
}

void ConfigFile::addValue_internal( ConfigSection *sect, const String &key, const String &value )
{
   MapIterator pos;
   ListElement *addPoint;
   ConfigEntry *entry = 0;

   // If there is already an entry, it's elegant to add the new value below the old ones.
   if ( sect->m_entries.find( &key, pos ) )
   {
      entry = *(ConfigEntry **) pos.currentValue();
      addPoint = (ListElement *) entry->m_values.end()->data();
   }
   else {
      // just add the value as the last entry of the section
      entry = new ConfigEntry;
      entry->m_entry = key;
      sect->m_entries.insert( &entry->m_entry, entry );
      addPoint = sect->m_additionPoint;
   }

   ConfigFileLine *line = new ConfigFileLine( ConfigFileLine::t_keyval,
         0,
         new String( key ),
         new String( value ) );

   if( addPoint == 0 )
   {
      m_lines.pushFront( line );
      addPoint = m_lines.begin();
      sect->m_additionPoint = addPoint;
   }
   else {
      m_lines.insertAfter( addPoint, line );
      addPoint = addPoint->next();
   }

   // we must put the ListElement holding the LINE where the value has been set
   entry->m_values.pushBack( addPoint );
}

bool ConfigFile::removeValue_internal( ConfigSection *sect, const String &key )
{
   MapIterator pos;
   ConfigEntry *entry;

   // we must find the first entry, if present.
   if ( sect->m_entries.find( &key, pos ) )
   {
      entry = *(ConfigEntry **) pos.currentValue();
   }
   else {
      return false;
   }

   // the first value of the entry must be changed; the other must be deletd.
   ListElement *valline = entry->m_values.begin();

   // then we must remove all the values
   while( valline != 0 )
   {
      ListElement *toBeDeleted = (ListElement *) valline->data();
      m_lines.erase( toBeDeleted );
      valline = valline->next();
   }

   // then remove the entry from the map
   sect->m_entries.erase( pos );
   return true;
}

void ConfigFile::clearMainSection()
{
   m_rootEntry.m_entries.clear();
   m_rootEntry.m_sectDecl = 0;
   m_rootEntry.m_additionPoint = 0;

   ListElement *line = m_lines.begin();
   while( line != 0 )
   {
      ConfigFileLine *lineData = (ConfigFileLine *) line->data();
      if ( lineData->m_type == ConfigFileLine::t_section )
         break;
      m_lines.popFront(); // will also destroy the data
      line = m_lines.begin();
   }
}


bool ConfigFile::removeSection( const String &key )
{
   MapIterator pos;
   if( ! m_sections.find( &key, pos ) )
      return false;

   ConfigSection *sect = *(ConfigSection **) pos.currentValue();
   ListElement *line = sect->m_sectDecl;
   if ( line != 0 )
   {
      line = m_lines.erase( line );
   }

   while( line != 0 )
   {
      ConfigFileLine *lineData = (ConfigFileLine *) line->data();
      if ( lineData->m_type == ConfigFileLine::t_section )
         break;
      line = m_lines.erase( line ); // will also destroy the data
   }

   m_sections.erase( pos );
   return true;
}

}

/* end of confparser_mod.cpp */

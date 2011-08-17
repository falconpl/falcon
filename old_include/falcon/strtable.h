/*
   FALCON - The Falcon Programming Language.
   FILE: strtable.h

   String table used in modules
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon Feb 14 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   String table used in modules
*/

#ifndef flc_strtable_H
#define flc_strtable_H

#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/genericvector.h>
#include <falcon/genericmap.h>
#include <falcon/basealloc.h>

namespace Falcon {

class Stream;
class ModuleLoader;

class FALCON_DYN_CLASS StringTable: public BaseAlloc
{
   GenericVector m_vector;
   Map m_map;
   Map m_intMap;
   char *m_tableStorage;
   uint32 m_internatCount;

   friend class ModuleLoader;

   // Non-const version of get is private
   String *getNonConst( uint32 id )
   {
      if ( id < m_vector.size() )
         return *(String **) m_vector.at( id );
      return 0;
   }

public:
   StringTable();
   StringTable( const StringTable &other );
   ~StringTable();

   void reserve( int32 size ) {
      m_vector.reserve( size );
   }

   int32 add( String *str );
   int32 findId( const String &str ) const;
   String *find( const String &source ) const;

   /** Skip the string table from a stream.
      This is useful when i.e. you have a string table embedded in a stream
      but you don't want to load it, for example because you want to use
      an external string table instead.
   */
   bool skip( Stream *in ) const;

   const String *get( uint32 id ) const
   {
      if ( id < m_vector.size() )
         return *(String **) m_vector.at( id );
      return 0;
   }

   const String *operator[]( int32 id ) {
      return *(String **) m_vector.at( id );;
   }

   int32 size() const { return m_vector.size(); }

   /** Save the string table in a stream.
      The string table is saved as a block, without using the serialization function
      of the Falcon::String objects. The block has a string table specific format,
      so that the load() function can load the whole block back in memory and then
      create each string in the table as a static non-zero terminated or static
      zero terminated string (in the proper encoding). None of the serialized
      string is re-created as bufferized, as the memory in which the string raw
      data resides is held internally in this object. Also, notice that all
      the strings and the relative raw data memory is destroyed with this object.

      The rationale for this behavior is that the StringTable object is meant
      to hold a specific set of strings that are related to some specific task.
      Usually, serialization and de-serialization of the string table occurs in
      module compilation and loading. As the vast majority of the  strings in
      the table will be accessed read only, to provide flexible storage at load
      time for all of the would be unefficient.

      The data block is aligned to a multiple of 4. I.e. if the function were
      to write 1438 bytes, it will actually deliver on the stream 1440 bytes,
      two of which being padding.

      The function never fails, but if the output stream has a failure
      the function doesn't detect it. The output stream status must be checked
      on exit.

      \param out the stream where to save the table.
      \return true on success, false on failure
   */
   bool save( Stream *out ) const;

   /** Restores a string table that was saved on a stream.
      For more details see save().
      \see save()
      \param in the input stream where the table must be loaded from
      \return true on success, false on failure (format error).
   */
   bool load( Stream *in );

   /** Saves a template file out of this string table.
      Template files are needed for internationalization.
      The template file will be written in an XML format. This function
      doesn't write the ?xml header of the xml file, as that
      requires the caller to know the encoding of the output stream.

      The caller should do it instead.

      \TODO Add encoding ID to common Stream interface.

      A template file contains all the strings of the table
      so that the compiler of a translation set can
      associate them with translation.
      \param out A Falcon::Stream for output.
      \param modName The name of the moule to be written in the template file.
      \param origLangCode the language code of this symbol table.
   */
   bool saveTemplate( Stream *out, const String &modName, const String &origLangCode ) const;

   /** Builds the table from a source file initialization.
      Useful to build static string tables in modules and
      in the engine.
      Provide the function with an array of char pointers,
      the last of which being 0; this will create a suitable
      module string table where the first string in the
      array has id 0, the second has id 1 and so on.
      \param table a vector of char * terminated by zero.
      \param bInternational the string table contains items to be internationalized.
   */
   void build( char **table, bool bInternational );

   /** Builds the table from a source file initialization.
      Useful to build static string tables in modules and
      in the engine.
      Provide the function with an array of char pointers,
      the last of which being 0; this will create a suitable
      module string table where the first string in the
      array has id 0, the second has id 1 and so on.
      \param table a vector of wchar_t * terminated by zero.
      \param bInternational the string table contains items to be internationalized.
   */
   void build( wchar_t **table, bool bInternational );

   /** Returns the count of international strings added to this symbol table.
      \note if this is zero, then the module writers shouldn't even create
      the template for the given module.
   */
   uint32 internatCount() const { return m_internatCount; }
};

}

#endif

/* end of strtable.h */

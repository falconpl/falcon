/*
   FALCON - The Falcon Programming Language.
   FILE: flc_deptab.h

   Dependency table declaration
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 8 2004

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef flc_DEPTAB_H
#define flc_DEPTAB_H

#include <falcon/string.h>
#include <falcon/genericmap.h>

/** \file
   Dependency table support for modules - header - .
*/

namespace Falcon {

class Module;
class Stream;

/** Class storing dependency data. */
class ModuleDepData: public BaseAlloc
{
   String m_modName;
   bool m_bPrivate;
   bool m_bFile;

public:
   ModuleDepData( const String modName, bool bPrivate = false, bool bFile = false ):
      m_modName( modName ),
      m_bPrivate( bPrivate ),
      m_bFile( bFile )
   {}

   const String &moduleName() const { return m_modName; }
   bool isPrivate() const { return m_bPrivate; }
   bool isFile() const { return m_bFile; }
   void setPrivate( bool mode ) { m_bPrivate = mode; }
};

/** Module dependency table.
   Actually it's just a string map supporting module-aware serialization.
   The strings are actually held in the module string table.
*/
class FALCON_DYN_CLASS DependTable: public Map
{
public:
   DependTable();
   ~DependTable();

   bool save( Stream *out ) const ;
   bool load( Module *mod, Stream *in );

   /** Adds a dependency to the table.
      This method creates a new dependency entry with a given
      dependency local alias, a physical or logical module name
      and a privacy setting.

      \note Strings must be pointers to Strings held in the module
      string table.

      \param alias The local module alias.
      \param name The logical or physical module name.
      \param bPrivate true if the module is private, false to honor its exports.
   */
   void addDependency( const String &alias, const String &name, bool bPrivate, bool bFile=false );

   /** Adds a dependency to the table.
      This version of the function adds a dependency with the same physical
      or logical name as the local alias.
   */
   void addDependency( const String &name, bool bPrivate = false, bool bFile = false ) {
      addDependency( name, name, bPrivate, bFile );
   }

   ModuleDepData *findModule( const String &name ) const
   {
      ModuleDepData **data = (ModuleDepData **) find( &name );
      if( data == 0 )
         return 0;

      return *data;
   }
};

}

#endif

/* end of flc_deptab.h */

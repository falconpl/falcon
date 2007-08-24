/*
   FALCON - The Falcon Programming Language.
   FILE: scriptdata.h
   $Id: scriptdata.h,v 1.3 2006/10/16 14:04:39 gian Exp $

    Defintions for script data dictionary.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: lun feb 13 2006
   Last modified because:

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
   In order to use this file in its compiled form, this source or
   part of it you have to read, understand and accept the conditions
   that are stated in the LICENSE file that comes boundled with this
   package.
*/

/** \file
   Header file needed for the testsuite application.
   Defintions for script data dictionary.
*/

#ifndef flc_scriptdata_H
#define flc_scriptdata_H

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/genericlist.h>
#include <map>

typedef std::map<Falcon::String, Falcon::String> t_stringMap;

namespace Falcon {

class ScriptData
{
   t_stringMap m_properties;
   int m_id;
   String m_filename;
public:
   ScriptData( const String &filename, int id=-1 );

   static void IdToIdCode( int id, String &code );
   static int IdCodeToId( const String &code );

   void setProperty( const String &name, const String &value );
   bool getProperty( const String &name, String &value ) const ;

   const String &filename() const { return m_filename; }
   const int id() const { return m_id; }
   void id( int num ) { m_id = num; }

   t_stringMap properties() { return m_properties; }
   const t_stringMap &properties() const { return m_properties; }
};

}

typedef std::map< int, Falcon::ScriptData *> t_idScriptMap;
typedef std::map< Falcon::String, Falcon::List> t_categoryScriptMap;

#endif

/* end of scriptdata.h */

/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_mod.h
 *
 * Helper/inner functions for DBI base.
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Mon, 13 Apr 2009 18:56:48 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */

#ifndef FALCON_DBI_MOD_H
#define FALCON_DBI_MOD_H

#include <dbiservice.h>
#include <falcon/string.h>

namespace Falcon {

int dbh_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value );
int dbh_realSqlExpand( VMachine *vm, DBIHandle *dbh, String &sql, int startAt=0 );
void dbh_escapeString( const String& input, String& value );
void dbh_throwError( const char* file, int line, int code, const String& desc );

void dbh_return_recordset( VMachine *vm, DBIRecordset *rec );

/** An utility used by many drivers to break connection strings.
   uid=
   pwd=
   host=
   port=
   db=
*/
class DBIConnParams
{

private:
   /** A Specification of a connection parameter.

      The parameter is a pair of a parameter name (as "uid" or "pwd") and
      of the output parameter, that is a pointer to a string that will be filled
      with the parameter name.
   */
   class Param
   {
   public:
      Param( const String& name, String& output ):
         m_name( name ),
         m_output( output ),
         m_szOutput( 0 ),
         m_cstrOut(0),
         m_pNext(0)
         {}

      Param( const String& name, String& output, const char** szOut ):
         m_name( name ),
         m_output( output ),
         m_szOutput( szOutput ),
         m_cstrOut(0),
         m_pNext(0)
         {}

      ~Param();

      /** Parses an input string.
       * @param value A string in format <name>=<value> or <name>=
       * @return True if the name matches (case insensitive) the name of this parameter.
       */
      bool parse( const String& value );

      String m_name;
      String &m_output;
      const char** m_szOutput;
      AutoCString* m_cstrOut;
      Param* m_pNext;
   };

   Param* m_pFirst;

   bool parsePart( const String& strPart );

protected:
   /** Function adding a parse parameter */
   void addParameter( const String& name, String& value );

   /** Function adding a parse parameter and its c-string value */
   void addParameter( const String& name, String& value, const char** szValue );

public:
   DBIConnParams();
   ~DBIConnParams();

   bool parse( const String& connStr );

   // Base parameters known by all systems.
   String m_sUser;
   String m_sPassword;
   String m_sHost;
   String m_sPort;
   String m_sDb;

   const char* m_szUser;
   const char* m_szPassword;
   const char* m_szHost;
   const char* m_szPort;
   const char* m_szDb;
};

}

#endif

/* end of dbi_mod.h */

/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_params.h

   Database Interface - Generic settings/parameters parser for DBI.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:16:00 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_DBI_PARAMS_H_
#define FALCON_DBI_PARAMS_H_

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>
#include <falcon/autocstring.h>

namespace Falcon {

/** An utility used by many drivers to break connection strings.
   Base class for more specific
*/
class DBIParams
{

protected:
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

      Param( const String& name, String& output, const char** szOutput ):
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

   DBIParams();

public:
   virtual ~DBIParams();

   /** Function adding a parse parameter */
   virtual void addParameter( const String& name, String& value );

   /** Function adding a parse parameter and its c-string value */
   virtual void addParameter( const String& name, String& value, const char** szValue );

   /** Parse the parameter=value string */
   virtual bool parse( const String& connStr );

   /** Utility to check for boolean values */
   static bool checkBoolean( const String& pvalue, bool &boolVar );
};


/** Parameter parser for the settings string.

    The setting string is used at DBIConnect,
    Connection.setOptions and Connection.tropen;
    they determine the behavior of db fetches, and may
    be implemented differently by different database drivers.

    Setting options are:

    - prefetch: number of records to be pre-fetched at each query. The value
                may be "all" to wholly fetch queries locally, "none" to prefetch
                none or an arbitrary number of rows to be read from the server.
                By default, it's "all".
    - autocommit: Performs a transaction commit after each sql command.
                  Can be "on" or "off"; it's "off" by default.
    - cursor: Number of records returned by a query that should trigger the creation of a
              server side cursor. Can be "none" to prevent creation of server
              side cursor (the default) "all" to always create a cursor or an arbitrary
              number to create a server side cursor only if the query returns at least
              the indicated number of rows.
    - strings: If "on", all the values are returned as a string. Can be useful if
               the engine provides this mode natively and if the recordset is needed
               just for dump on an output device. Using this option in this case will
               reduce unneeded transformations into Falcon data and then into the
               external representationss

    After a complete local prefetch, all the records
    are moved to the client, so it's possible to issue another query returning a different
    recordset. If moving all the returned rows to the client is not feasible, but it's still
    necessary to fetch rows from a query and performing other queries based on the retrieved
    results, then it's necessary to create a cursor on the server side. Many SQL engines allow
    to specify to open the cursor directly on the SQL statement, but this option is given
    to provide the user with the ability to access this feature without using engine-specific
    SQL.

    Notice that server-side cursor and complete local prefetch are alternative methods
    to make the recordset consistent across different queries in the same transaction,
    so it's pretty useless to use both of them at the same time.

    This class transforms the given options in C variables that can easily be read by
    the drivers. Also, drivers can add their own option parsing code by overriding the
    DBIConnection::parseOptions method.
*/

class DBISettingParams: public DBIParams
{
private:
   String m_sCursor;
   String m_sAutocommit;
   String m_sPrefetch;
   String m_sFetchStrings;

   static const bool defaultAutocommit = true;
   static const int defaultCursor = -1;
   static const int defaultPrefetch = -1;
   static const bool defaultFetchStrings = false;

public:
   DBISettingParams();
   DBISettingParams( const DBISettingParams & other );
   virtual ~DBISettingParams();

   /** Specific parse analizying the options */
   virtual bool parse( const String& connStr );

   /** True if the transaction should be autocommit, false otherwise. */
   bool m_bAutocommit;

   /** Cursor invocation treshold.
    * Will be -1 if cursor should be never used, 0 if always used,
    * a positive number for a given treshold.
    */
   int64 m_nCursorThreshold;

   /** Number of rows to be pre-fetched after queries.
      Will be -1 if recordset must be completely prefetched, 0 to disable pre-fetching
      and a positive number to ask for prefetching of that many rows.
   */
   int64 m_nPrefetch;

   /** True if the transaction should be autocommit, false otherwise. */
   bool m_bFetchStrings;

};


/** An utility used by many drivers to break connection strings.

   This is just an utility class that a driver may use to break connection
   strings into parameters for its DB system. A driver is free not to use
   it if, for example, it thinks it should pass the string as-is to the
   underlying DB system.

   The format of the string respects ODBC standards ("parameter=value;..."),
   so the connection stings for systems that don't rely on that to initialize the connections
   (as, for example SQLlite and MySQL) looks like ODBC connection, giving
   a flavor of portability of the connection string across different system.

   The utility works like this: a set of parameters are declared via the
   method addParameter(); other than the parameter name, this method allows
   to declare a parameter value, that is, a locally defined string where
   the parameter value will be placed if found, and optionally a C zero terminated
   string where to put the AutoCString result (converting the string in utf-8),
   in case the driver needs this  feature.

   If the required parameter is not found in the parsed string, the string is left
   empty, and the C ZTstring, if provided, is set to a null pointer. If it's found,
   but empty, then the string is set to a pair of double quotes (""), and the c ZTString,
   if provided, is set to an empty string.

   To ensure a minimal coherence across different drivers, and to reduce the effort of the
   implementers, a minimal set of common parameters are added by the constructor of this
   class. The variables where the parameter values are placed are provided by the class
   as well. Namely:

   - uid= The user id for the connection (values placed in m_sUser and m_szUser).
   - pwd= Password for the connection (values placed in m_sPassword and m_szPassword).
   - db= Name of the DB to open (values in m_sDb and m_szDb).
   - host= Host where to perform the connection (values placed in m_sHost and m_szHost).
   - port= TCP Port where the server is listening (values in m_sPort and m_szPort).
   - create= set to "always" or "cond" to create the database or try to create it in case it doesn't exist.
*/
class DBIConnParams: public DBIParams
{
public:
   DBIConnParams( bool bNoDefaults = false );
   virtual ~DBIConnParams();

   // Base parameters known by all systems.
   String m_sUser;
   String m_sPassword;
   String m_sHost;
   String m_sPort;
   String m_sDb;
   String m_sCreate;

   const char* m_szUser;
   const char* m_szPassword;
   const char* m_szHost;
   const char* m_szPort;
   const char* m_szDb;
   const char* m_szCreate;
};

}

#endif

/* dbi_params.h */

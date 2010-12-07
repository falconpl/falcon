/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_params.cpp

   DBI service that DBI drivers inherit from and implement
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 15 May 2010 15:40:39 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_params.h>

namespace Falcon
{

//============================================================
// Generic parameter parser
//============================================================
DBIParams::DBIParams():
   m_pFirst(0)
{
}

DBIParams::~DBIParams()
{
   Param* p = m_pFirst;
   while( p != 0 )
   {
      Param* old = p;
      p = p->m_pNext;
      delete old;
   }

   m_pFirst = 0;
}

bool DBIParams::checkBoolean( const String& pvalue, bool &boolVar )
{
   if( pvalue.compareIgnoreCase("on") == 0 )
   {
      boolVar = true;
   }
   else if( pvalue.compareIgnoreCase("off") == 0 )
   {
      boolVar = false;
   }
   else if ( pvalue != "" && pvalue != "\"\"" )
   {
      return false;
   }

   return true;
}


void DBIParams::addParameter( const String& name, String& value )
{
   Param* p = new Param( name, value );
   p->m_pNext = m_pFirst;
   m_pFirst = p;
}


void DBIParams::addParameter( const String& name, String& value, const char **szValue )
{
   Param* p = new Param( name, value, szValue );
   p->m_pNext = m_pFirst;
   m_pFirst = p;
}


bool DBIParams::parse( const String& connStr )
{
   uint32 pos = 0;

   // find next ";"
   uint32 pos1 = connStr.find(";");

   do
   {
      pos1 = connStr.find( ";", pos );
      String part = connStr.subString( pos, pos1 );
      pos = pos1 + 1;

      if ( ! parsePart( part ) )
      {
         return false;
      }
   }
   while( pos1 != String::npos );

   return true;
}


bool DBIParams::parsePart( const String& strPart )
{
   // Break the record <key>=<value>
   uint32 pos = strPart.find( "=" );
   if ( pos == String::npos )
      return false;

   String key = strPart.subString(0,pos);
   key.trim();

   Param* p = m_pFirst;
   while( p != 0 )
   {
      if ( p->m_name.compareIgnoreCase( key ) == 0 )
      {
         // Found!
         p->m_output = strPart.subString( pos + 1 );
         // use this convention to declare an empty, but given value.
         if( p->m_output == "" )
         {
            p->m_output = "''";
            if( p->m_szOutput != 0 )
               *p->m_szOutput = ""; // given but empty
         }
         else
         {
            if( p->m_szOutput != 0 )
            {
               p->m_cstrOut = new AutoCString( p->m_output );
               *p->m_szOutput = p->m_cstrOut->c_str();
            }
         }

         return true;
      }
      p = p->m_pNext;
   }

   // not found... it's an invalid key
   return false;
}


DBIParams::Param::~Param()
{
   delete m_cstrOut;
}

//============================================================
// Settings parameter parser
//============================================================
DBISettingParams::DBISettingParams():
      m_bAutocommit( defaultAutocommit ),
      m_nCursorThreshold( defaultCursor ),
      m_nPrefetch( defaultPrefetch ),
      m_bFetchStrings( defaultFetchStrings )
{
   addParameter( "autocommit", m_sAutocommit );
   addParameter( "cursor", m_sCursor );
   addParameter( "prefetch", m_sPrefetch );
   addParameter( "strings", m_sFetchStrings );
}

DBISettingParams::DBISettingParams( const DBISettingParams & other):
   m_bAutocommit( other.m_bAutocommit ),
   m_nCursorThreshold( other.m_nCursorThreshold ),
   m_nPrefetch( other.m_nPrefetch ),
   m_bFetchStrings( other.m_bFetchStrings )
{
   // we don't care about the parameter parsing during the copy.
}


DBISettingParams::~DBISettingParams()
{
}


bool DBISettingParams::parse( const String& connStr )
{
   if( ! DBIParams::parse(connStr) )
   {
      return false;
   }

   if ( ! checkBoolean( m_sAutocommit, m_bAutocommit ) ) return false;
   if ( ! checkBoolean( m_sFetchStrings, m_bFetchStrings ) ) return false;

   if( m_sPrefetch.compareIgnoreCase("all") == 0 )
   {
      m_nPrefetch = -1;
   }
   else if( m_sPrefetch.compareIgnoreCase("none") == 0 )
   {
      m_nPrefetch = 0;
   }
   else if ( m_sPrefetch != "" && m_sPrefetch != "\"\"" )
   {
      if ( ! m_sPrefetch.parseInt( m_nPrefetch ) )
         return false;
   }

   if( m_sCursor.compareIgnoreCase("none") == 0 )
   {
      m_nCursorThreshold = -1;
   }
   else if( m_sCursor.compareIgnoreCase("all") == 0 )
   {
      m_nCursorThreshold = 0;
   }
   else if ( m_sCursor != "" && m_sCursor != "\"\"" )
   {
      if ( ! m_sCursor.parseInt( m_nCursorThreshold ) )
         return false;
   }

   return true;
}

//============================================================
// Connection parameter parser
//============================================================

DBIConnParams::DBIConnParams( bool bNoDef ):
      m_szUser(0),
      m_szPassword(0),
      m_szHost(0),
      m_szPort(0),
      m_szDb(0)
{
   if( ! bNoDef )
   {
      // add the default parameters
      addParameter( "uid", m_sUser, &m_szUser );
      addParameter( "pwd", m_sPassword, &m_szPassword );
      addParameter( "db", m_sDb, &m_szDb );
      addParameter( "port", m_sPort, &m_szPort );
      addParameter( "host", m_sHost, &m_szHost );
      addParameter( "create", m_sCreate, &m_szCreate );
   }
}


DBIConnParams::~DBIConnParams()
{
}

}

/* end of dbi_params.cpp */


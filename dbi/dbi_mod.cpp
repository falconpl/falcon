/*
 * FALCON - The Falcon Programming Language.
 * FILE: dbi_mod.cpp
 *
 *
 * -------------------------------------------------------------------
 * Author: Giancarlo Niccolai and Jeremy Cowgar
 * Begin: Mon, 13 Apr 2009 18:56:48 +0200
 *
 * -------------------------------------------------------------------
 * (C) Copyright 2007: the FALCON developers (see list in AUTHORS file)
 *
 * See LICENSE file for licensing details.
 */


#include "dbi_mod.h"
#include "dbi.h"
#include <dbiservice.h>

/******************************************************************************
 * Local Helper Functions - DBH database handle
 *****************************************************************************/
namespace Falcon
{

int dbh_itemToSqlValue( DBIHandle *dbh, const Item *i, String &value )
{
   switch( i->type() ) {
      case FLC_ITEM_BOOL:
         value = i->asBoolean() ? "TRUE" : "FALSE";
         return 1;

      case FLC_ITEM_INT:
         value.writeNumber( i->asInteger() );
         return 1;

      case FLC_ITEM_NUM:
         value.writeNumber( i->asNumeric(), "%f" );
         return 1;

      case FLC_ITEM_STRING:
         dbh_escapeString( *i->asString(), value );
         value.prepend( "'" );
         value.append( "'" );
         return 1;

      case FLC_ITEM_OBJECT: {
            CoreObject *o = i->asObject();
            //vm->itemToString( value, ??? )
            if ( o->derivedFrom( "TimeStamp" ) ) {
               TimeStamp *ts = (TimeStamp *) o->getUserData();
               ts->toString( value );
               value.prepend( "'" );
               value.append( "'" );
               return 1;
            }
            return 0;
         }

      case FLC_ITEM_NIL:
         value = "NULL";
         return 1;

      default:
         return 0;
   }
}



void dbh_return_recordset( VMachine *vm, DBIRecordset *rec )
{
   Item *rsclass = vm->findWKI( "%DBIRecordset" );
   fassert( rsclass != 0 && rsclass->isClass() );

   CoreObject *oth = rsclass->asClass()->createInstance();
   oth->setUserData( rec );
   vm->retval( oth );
}


void dbh_escapeString( const String& input, String& value )
{
   uint32 len = input.length();
   uint32 pos = 0;
   value.reserve( len + 8 );

   while( pos < len )
   {
      uint32 chr = input.getCharAt(pos);
      switch( chr )
      {
         case '\\':
            value.append(chr);
            value.append(chr);
            break;

         case '\'':
            value.append( '\\' );
            value.append( '\'' );
            break;

         case '"':
            value.append( '\\' );
            value.append( '"' );
            break;

         default:
            value.append( chr );
      }
      ++pos;
   }

}

void dbh_throwError( const char* file, int line, int code, const String& desc )
{
   VMachine* vm = VMachine::getCurrent();

   if ( vm != 0 )
   {
      int msgId = code - FALCON_DBI_ERROR_BASE - 1;

      throw new DBIError( ErrorParam( code, line )
             .desc( vm->moduleString( msgId ) )
             .module( file )
             .extra( desc )
          );
   }
   else
   {
      throw new DBIError( ErrorParam( code, line )
         .desc( "Unknown error code" )
         .module( file )
         .extra( desc )
      );
   }
}

//============================================================
// Connection parameter parser
//============================================================
DBIConnParams::DBIConnParams():
   m_pFirst(0)
{
   // add the default parameters
   addParameter( "uid", m_sUser, &m_szUser );
   addParameter( "pwd", m_sPassword, &m_szPassword );
   addParameter( "db", m_sDb, &m_szDb );
   addParameter( "port", m_sPort, &m_szPort );
   addParameter( "host", m_sHost, &m_szHost );
}


DBIConnParams::~DBIConnParams()
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


void DBIConnParams::addParameter( const String& name, String& value )
{
   Param* p = new Param( name, value );
   p->m_pNext = m_pFirst;
   m_pFirst = p;
}


void DBIConnParams::addParameter( const String& name, String& value, const char **szValue )
{
   Param* p = new Param( name, value, szValue );
   p->m_pNext = m_pFirst;
   m_pFirst = p;
}


bool DBIConnParams::parse( const String& connStr )
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


bool DBIConnParams::parsePart( const String& strPart )
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
   }

   // not found... it's an invalid key
   return false;
}


DBIConnParams::Param::~Param()
{
   delete m_cstrOut;
}

}

/* end of dbi_mod.cpp */
